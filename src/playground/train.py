"""One-epoch Gemma 4 E4B tool-call LoRA at a 32k token cap.

Examples are rendered with Gemma's chat template, not the plain JSON transcript.
The adapter is data/checkpoints/e4b-32k-tools. Batch size 1 stays near 160GB.
Batch size 2 is about 290GB, so it is refused under the 200GB ceiling.
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

from playground.config import ROOT
from playground.gemma_turns import TOOL_TRAIN_EXAMPLES, build_checked_tool_train_set
from playground.harness.tools import openai_tools
from playground.probe import (
    MEMORY_CEILING_BYTES,
    MODEL_NAME,
    CountTokens,
    enforce_metal_ceiling,
    load_teacher_rows,
    memory_pressure,
    pack_example,
    phys_footprint,
)

TRAIN_CONTEXT = 32768
MAX_BATCH_SIZE = 1
CHECKPOINT_DIR = ROOT / "data" / "checkpoints" / "e4b-32k-lora"
TOOLS_CHECKPOINT_DIR = ROOT / "data" / "checkpoints" / "e4b-32k-tools"


def validate_train_args(context: int, batch_size: int) -> None:
    if context != TRAIN_CONTEXT:
        raise SystemExit(
            f"This train is capped at {TRAIN_CONTEXT} tokens. Refusing context {context}."
        )
    if batch_size != MAX_BATCH_SIZE:
        raise SystemExit(
            f"Batch size {batch_size} is not allowed. Batch size 1 stays near 160GB. "
            "Batch size 2 is about 290GB and would cross the 200GB ceiling."
        )


def build_train_set(rows: list[dict], limit: int, count_tokens: CountTokens) -> tuple[list[dict], int]:
    """Train rows only, each packed to ``limit`` tokens. Holdout rows are skipped."""
    packed: list[dict] = []
    skipped = 0
    for row in rows:
        if row.get("split") != "train":
            skipped += 1
            continue
        item = pack_example(row, limit, count_tokens)
        if item is None or item["tokens"] > limit:
            skipped += 1
            continue
        packed.append(item)
    if not packed:
        raise SystemExit("No train episodes fit under the 32k cap.")
    return packed, skipped


def resolve_tools_checkpoint(checkpoint: Path | None) -> Path:
    """The new adapter is e4b-32k-tools. Never overwrite e4b-32k-lora."""
    dest = checkpoint or TOOLS_CHECKPOINT_DIR
    if dest.resolve() == CHECKPOINT_DIR.resolve():
        raise SystemExit(
            "Refusing to write over data/checkpoints/e4b-32k-lora. "
            "The tool-call adapter goes to data/checkpoints/e4b-32k-tools."
        )
    return dest


def chat_template_renderer(tokenizer):
    """Render with Gemma's template. The loss string must contain its tool-call tokens."""
    tools = openai_tools()

    def render(messages: list[dict]) -> str:
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
        )
        if not isinstance(text, str):
            raise SystemExit("apply_chat_template did not return text.")
        if any(message.get("tool_calls") for message in messages) and "<|tool_call>call:" not in text:
            raise SystemExit(
                "Stopping. apply_chat_template did not emit <|tool_call>call:. "
                "Refusing to train a plain-text adapter."
            )
        return text

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return render, count


def run_train_cli(
    context: int,
    batch_size: int,
    *,
    examples: int = TOOL_TRAIN_EXAMPLES,
    checkpoint: Path | None = None,
) -> None:
    """Train the tool-call LoRA. This does not resume data/checkpoints/e4b-32k-lora."""
    validate_train_args(context, batch_size)
    from playground.probe import _require_unsloth

    dest = resolve_tools_checkpoint(checkpoint)
    rows = load_teacher_rows()
    tokenizer = _load_chat_tokenizer()
    render, count_tokens = chat_template_renderer(tokenizer)
    packed, skipped = build_checked_tool_train_set(rows, context, count_tokens, render, n=examples)
    tokens = sum(item["tokens"] for item in packed)
    print(
        f"Tool-call train set: {len(packed)} episodes, {tokens} tokens, skipped {skipped}, "
        f"batch_size={batch_size}",
        flush=True,
    )
    _require_unsloth()
    dest.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    peak = {"bytes": phys_footprint()}
    outcome = {"status": "finished"}

    def watch() -> None:
        while not stop.wait(0.5):
            enforce_metal_ceiling()
            used = phys_footprint()
            peak["bytes"] = max(peak["bytes"], used)
            if used >= MEMORY_CEILING_BYTES:
                outcome["status"] = "killed"
                _write_status(
                    "killed", examples=len(packed), tokens=tokens, peak_bytes=used, dest=dest
                )
                os.kill(os.getpid(), signal.SIGTERM)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        _unsloth_train([item["text"] for item in packed], context, batch_size, dest)
    finally:
        stop.set()
        after = phys_footprint()
        peak["bytes"] = max(peak["bytes"], after)
        if outcome["status"] != "killed":
            _write_status(
                "finished", examples=len(packed), tokens=tokens, peak_bytes=peak["bytes"], dest=dest
            )
        print(
            f"Saved {dest} status={outcome['status']} "
            f"peak_bytes={peak['bytes']} pressure={memory_pressure()}",
            flush=True,
        )


def _load_chat_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "The train tokenizer is not installed. On the Studio run: uv sync --group studio"
        ) from exc
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def _unsloth_train(texts: list[str], max_seq_length: int, batch_size: int, dest: Path) -> None:
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

    args = TrainingArguments(
        output_dir=str(dest),
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    dataset = [{"text": text} for text in texts]
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=args,
        )
    enforce_metal_ceiling()
    try:
        from transformers import TrainerCallback
    except ImportError:
        TrainerCallback = None  # type: ignore[misc, assignment]
    if TrainerCallback is not None:
        class _Ceiling(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                enforce_metal_ceiling()

            def on_step_begin(self, args, state, control, **kwargs):
                enforce_metal_ceiling()

        trainer.add_callback(_Ceiling())
    enforce_metal_ceiling()
    trainer.train()
    trainer.save_model(str(dest))


def _write_status(
    status: str, *, examples: int, tokens: int, peak_bytes: int, dest: Path = TOOLS_CHECKPOINT_DIR
) -> None:
    import json

    dest.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "context": TRAIN_CONTEXT,
        "examples": examples,
        "tokens": tokens,
        "peak_bytes": peak_bytes,
        "ceiling_bytes": MEMORY_CEILING_BYTES,
    }
    (dest / "train-status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
