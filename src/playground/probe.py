"""One-step Gemma 4 E4B LoRA memory probes for the Mac Studio.

The probe does not start a full train. It packs one teacher trace, takes one
bf16 LoRA step through Unsloth, and records unified memory. The next context
is refused when the previous probe was killed or crossed 200GB.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Callable

from playground.config import ROOT, TRACES_DIR
from playground.traces import read_jsonl

MEMORY_CEILING_BYTES = 200 * 1024 ** 3
# MLX cache on top of a 32k step (~155GB) must still fit under the ceiling.
METAL_CACHE_LIMIT_BYTES = 16 * 1024 ** 3
PROBE_CONTEXTS = (16384, 32768, 98304)
PROBE_NAMES = {16384: "probe-16k", 32768: "probe-32k", 98304: "probe-96k"}
TEACHER_TRACE_PARTS = (
    TRACES_DIR / "qwen3-8-flash-next.part1.jsonl",
    TRACES_DIR / "qwen3-8-flash-next.part2.jsonl",
)
TEACHER_TRACE = TRACES_DIR / "qwen3-8-flash-next.jsonl"
PROBES_DIR = ROOT / "data" / "probes"
MODEL_NAME = "unsloth/gemma-4-E4B-it"

CountTokens = Callable[[str], int]


def load_teacher_rows() -> list[dict]:
    """Read the teacher set. Parts stay under GitHub's 100MB file limit."""
    parts = [path for path in TEACHER_TRACE_PARTS if path.is_file()]
    if parts:
        rows: list[dict] = []
        for path in parts:
            rows.extend(read_jsonl(path))
        return rows
    if TEACHER_TRACE.is_file():
        return read_jsonl(TEACHER_TRACE)
    raise SystemExit("No teacher traces. Expected data/traces/qwen3-8-flash-next.part1.jsonl and part2.jsonl.")


def probe_path(context: int, probes_dir: Path = PROBES_DIR) -> Path:
    name = PROBE_NAMES.get(context)
    if name is None:
        raise SystemExit(f"Unsupported probe context {context}. Choose 16384, 32768, or 98304.")
    return probes_dir / f"{name}.json"


def train_episodes(rows: list[dict]) -> list[dict]:
    """Train episodes that contain a real read_file page, longest first."""
    candidates = [row for row in rows if row.get("split") == "train" and _read_pages(row)]
    return sorted(candidates, key=lambda row: (row.get("token_estimate") or 0, row.get("id") or ""), reverse=True)


def select_train_episode(rows: list[dict]) -> dict:
    """Longest train episode that still has a real read_file page."""
    candidates = train_episodes(rows)
    if not candidates:
        raise SystemExit("No train episode with a read_file page in the teacher traces.")
    return candidates[0]


def pack_probe_example(rows: list[dict], limit: int, count_tokens: CountTokens) -> dict | None:
    """Longest train episode that still fits under ``limit`` after trimming pages."""
    for episode in train_episodes(rows):
        packed = pack_example(episode, limit, count_tokens)
        if packed is not None:
            return packed
    return None


def build_messages(episode: dict, page_texts: list[str]) -> list[dict]:
    """Chat text for the step. Thinking stays out of every message."""
    pages = list(page_texts)
    messages = [{"role": "user", "content": episode.get("commission") or ""}]
    cursor = 0
    for item in episode.get("tool_rounds") or []:
        name = item.get("name") or ""
        arguments = dict(item.get("arguments") or {})
        messages.append(
            {"role": "assistant", "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}
        )
        result = dict(item.get("result") or {})
        if name == "read_file":
            result["content"] = pages[cursor] if cursor < len(pages) else ""
            cursor += 1
        messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    messages.append({"role": "assistant", "content": episode.get("report") or ""})
    return messages


def render_messages(messages: list[dict]) -> str:
    return "\n".join(f"{message['role']}\n{message['content']}" for message in messages)


def pack_example(episode: dict, limit: int, count_tokens: CountTokens) -> dict | None:
    """Fit one example under ``limit`` using real read_file text. Never pad.

    Returns None when even the skeleton, with no page text, is over the limit
    or when a 96k pack cannot hold any real page text.
    """
    pages = _read_pages(episode)
    if limit >= 98304:
        kept = _pack_pages(episode, pages, limit, count_tokens)
    else:
        kept = _trim_pages(episode, pages, limit, count_tokens)
    if kept is None:
        return None
    messages = build_messages(episode, kept)
    text = render_messages(messages)
    return {
        "text": text,
        "tokens": count_tokens(text),
        "thinking": episode.get("thinking") or [],
        "pages_used": sum(1 for page in kept if page),
    }


def assert_previous_probe(context: int, probes_dir: Path = PROBES_DIR) -> None:
    index = PROBE_CONTEXTS.index(context)
    if index == 0:
        return
    previous = PROBE_CONTEXTS[index - 1]
    path = probe_path(previous, probes_dir)
    if not path.is_file():
        raise SystemExit(f"Refusing {PROBE_NAMES[context]}. {path.name} does not exist yet.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    peak = int(payload.get("peak_bytes") or 0)
    if payload.get("status") != "finished" or peak >= MEMORY_CEILING_BYTES:
        raise SystemExit(
            f"Refusing {PROBE_NAMES[context]}. {path.name} status={payload.get('status')} peak_bytes={peak}."
        )


def run_probe_cli(context: int) -> None:
    assert_previous_probe(context)
    rows = load_teacher_rows()
    tokenizer = _load_tokenizer()
    packed = pack_probe_example(rows, context, tokenizer)
    path = probe_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    if packed is None:
        _write_probe(
            path,
            context=context,
            tokens=0,
            before_bytes=0,
            peak_bytes=0,
            after_bytes=0,
            status="dropped",
            memory_pressure=None,
        )
        print(f"Dropped {path.name}. Could not pack a real example under {context} tokens.", flush=True)
        return
    _require_unsloth()
    before = phys_footprint()
    peak = {"bytes": before, "pressure": memory_pressure()}
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.5):
            enforce_metal_ceiling()
            used = phys_footprint()
            peak["bytes"] = max(peak["bytes"], used)
            peak["pressure"] = memory_pressure()
            if used >= MEMORY_CEILING_BYTES:
                _write_probe(
                    path,
                    context=context,
                    tokens=packed["tokens"],
                    before_bytes=before,
                    peak_bytes=used,
                    after_bytes=used,
                    status="killed",
                    memory_pressure=peak["pressure"],
                )
                os.kill(os.getpid(), signal.SIGTERM)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    status = "finished"
    try:
        unsloth_one_step(packed["text"], context)
    except Exception:
        status = "killed"
        raise
    finally:
        stop.set()
        after = phys_footprint()
        peak["bytes"] = max(peak["bytes"], after)
        _write_probe(
            path,
            context=context,
            tokens=packed["tokens"],
            before_bytes=before,
            peak_bytes=peak["bytes"],
            after_bytes=after,
            status=status,
            memory_pressure=peak["pressure"],
        )
        print(
            f"{path.name} status={status} tokens={packed['tokens']} "
            f"before={before} peak={peak['bytes']} after={after}",
            flush=True,
        )


def unsloth_one_step(text: str, max_seq_length: int) -> None:
    """One bf16 LoRA step. Exits before loading weights if Unsloth is missing."""
    _require_unsloth()
    from unsloth import FastModel
    from transformers import TrainingArguments

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        full_finetuning=False,
    )
    lora = dict(
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    try:
        model = FastModel.get_peft_model(model, finetune_audio_layers=False, **lora)
    except TypeError:
        model = FastModel.get_peft_model(model, **lora)
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=[{"text": text}],
        args=TrainingArguments(
            output_dir=str(PROBES_DIR / "scratch"),
            per_device_train_batch_size=1,
            max_steps=1,
            learning_rate=2e-4,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        ),
    )
    enforce_metal_ceiling()
    trainer.train()


_ceiling_announced = False


def _mlx_set(mx, name: str, value: int):
    fn = getattr(mx, name, None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), name, None)
    if fn is None:
        return None
    try:
        return fn(value)
    except TypeError:
        fn(value)
        return None


def enforce_metal_ceiling() -> None:
    """Hard-cap MLX wired memory. A SIGTERM cannot stop an in-flight Metal allocation.

    Unsloth sets ``mx.set_wired_limit`` to most of the machine after import.
    The ``mx.metal.set_*`` names are deprecated and do not override that.
    """
    global _ceiling_announced
    try:
        import mlx.core as mx
    except ImportError:
        return
    limit = MEMORY_CEILING_BYTES
    cache = min(limit, METAL_CACHE_LIMIT_BYTES)
    previous_wired = _mlx_set(mx, "set_wired_limit", limit)
    _mlx_set(mx, "set_memory_limit", limit)
    _mlx_set(mx, "set_cache_limit", cache)
    if isinstance(previous_wired, int) and previous_wired > limit:
        print(
            f"MLX wired limit was {previous_wired / 1024 ** 3:.2f}GB, reset to {limit / 1024 ** 3:.0f}GB.",
            flush=True,
        )
        _ceiling_announced = True
    elif not _ceiling_announced:
        _ceiling_announced = True
        print(
            f"MLX wired limit set to {limit / 1024 ** 3:.0f}GB "
            f"(cache {cache / 1024 ** 3:.0f}GB).",
            flush=True,
        )


def phys_footprint() -> int:
    """Process physical footprint. On Apple Silicon this includes Metal."""
    if sys.platform == "darwin":
        measured = _darwin_phys_footprint()
        if measured is not None:
            return measured
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def memory_pressure() -> str | None:
    import subprocess

    try:
        proc = subprocess.run(["memory_pressure", "-Q"], capture_output=True, text=True, timeout=2, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (proc.stdout or proc.stderr or "").strip()
    return text or None


def _require_unsloth() -> None:
    try:
        import unsloth  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Unsloth is not installed. On the Studio run: uv sync --group studio"
        ) from exc


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The probe tokenizer is not installed. On the Studio run: uv sync --group studio"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def _read_pages(episode: dict) -> list[str]:
    pages = []
    for item in episode.get("tool_rounds") or []:
        if item.get("name") != "read_file":
            continue
        content = (item.get("result") or {}).get("content") or ""
        pages.append(content if isinstance(content, str) else "")
    return pages


def _trim_pages(episode: dict, pages: list[str], limit: int, count_tokens: CountTokens) -> list[str] | None:
    kept = list(pages)
    while kept and count_tokens(render_messages(build_messages(episode, kept))) > limit:
        for index in range(len(kept) - 1, -1, -1):
            if kept[index]:
                kept[index] = ""
                break
        else:
            break
    if count_tokens(render_messages(build_messages(episode, kept))) > limit:
        return None
    return kept


def _pack_pages(episode: dict, pages: list[str], limit: int, count_tokens: CountTokens) -> list[str] | None:
    kept = [""] * len(pages)
    if count_tokens(render_messages(build_messages(episode, kept))) > limit:
        return None
    used_real = False
    for index, page in enumerate(pages):
        if not page:
            continue
        trial = list(kept)
        trial[index] = page
        if count_tokens(render_messages(build_messages(episode, trial))) <= limit:
            kept = trial
            used_real = True
            continue
        lo, hi = 0, len(page)
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            trial[index] = page[:mid]
            if count_tokens(render_messages(build_messages(episode, trial))) <= limit:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best:
            kept[index] = page[:best]
            used_real = True
        break
    if not used_real:
        return None
    return kept


def _write_probe(
    path: Path,
    *,
    context: int,
    tokens: int,
    before_bytes: int,
    peak_bytes: int,
    after_bytes: int,
    status: str,
    memory_pressure: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": PROBE_NAMES[context],
        "context": context,
        "tokens": tokens,
        "before_bytes": before_bytes,
        "peak_bytes": peak_bytes,
        "after_bytes": after_bytes,
        "status": status,
        "memory_pressure": memory_pressure,
        "ceiling_bytes": MEMORY_CEILING_BYTES,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _darwin_phys_footprint() -> int | None:
    import ctypes

    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    task_info = libc.task_info
    task_info.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint),
    ]
    task_info.restype = ctypes.c_int
    task = libc.mach_task_self()
    # phys_footprint sits at byte 144 of task_vm_info on current macOS.
    raw = ctypes.create_string_buffer(512)
    count = ctypes.c_uint(512 // 4)
    kern_success = task_info(task, 22, raw, ctypes.byref(count))
    if kern_success != 0 or count.value < 40:
        return None
    return int.from_bytes(raw.raw[144:152], byteorder=sys.byteorder)
