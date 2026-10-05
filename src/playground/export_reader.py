"""Fuse the 300-step reader LoRA, DWQ-quantize it, and score the 4-bit model.

The Studio already has the base weights. This command uses that local snapshot
and does not download Gemma again.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from playground.config import ROOT, settings
from playground.subswe import F16_MODEL, LORA_MODEL, READER_4BIT_MODEL

BASE_MODEL = "unsloth/gemma-4-E4B-it"
BASE_COMMIT = "4e22d7e59e078e63a14f351efdc5232ed366b621"
ADAPTER_DIR = ROOT / "data" / "checkpoints" / "e4b-32k-tools-300"
BF16_DIR = ROOT / "data" / "models" / "gemma-4-e4b-reader-bf16"
FOUR_BIT_DIR = ROOT / "data" / "models" / "gemma-4-e4b-reader-4bit"


def check_adapter(path: Path) -> None:
    """The fuse has to use the Gemma this LoRA was trained on."""
    config_path = path / "adapter_config.json"
    weights = path / "adapters.safetensors"
    if not config_path.is_file() or not weights.is_file():
        raise SystemExit(f"LoRA adapter not found: {path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config.get("base_model_name_or_path")
    if base != BASE_MODEL:
        raise SystemExit(f"Adapter base is {base!r}. Expected {BASE_MODEL}.")
    commit = config.get("base_model_commit_hash") or config.get("base_model_revision")
    if commit and commit != BASE_COMMIT:
        raise SystemExit(f"Adapter commit is {commit}. Expected {BASE_COMMIT}.")


def local_gemma_snapshot(cache: Path) -> str:
    """Use the downloaded Gemma snapshot. Do not contact Hugging Face."""
    repo = cache / "models--unsloth--gemma-4-E4B-it"
    snapshots = repo / "snapshots"
    pinned = snapshots / BASE_COMMIT
    if (pinned / "config.json").is_file():
        return str(pinned)
    main_ref = repo / "refs" / "main"
    if main_ref.is_file():
        name = main_ref.read_text(encoding="utf-8").strip()
        pointed = snapshots / name
        if (pointed / "config.json").is_file():
            print(
                f"Using cached {BASE_MODEL} snapshot {name}. "
                f"Commit {BASE_COMMIT} is not a separate local snapshot.",
                flush=True,
            )
            return str(pointed)
    found = []
    if snapshots.is_dir():
        found = [path for path in snapshots.iterdir() if (path / "config.json").is_file()]
    if len(found) == 1:
        print(
            f"Using cached {BASE_MODEL} snapshot {found[0].name}.",
            flush=True,
        )
        return str(found[0])
    raise SystemExit(
        f"{BASE_MODEL} is not in the local Hugging Face cache at {repo}."
    )


def resolve_cached_base() -> str:
    """Return the local snapshot. Refuse to download when the model is missing."""
    from huggingface_hub.constants import HF_HUB_CACHE

    return local_gemma_snapshot(Path(HF_HUB_CACHE))


def fuse_bf16(adapter: Path, dest: Path) -> None:
    """Merge the LoRA with Unsloth. mlx-vlm has no fuse module."""
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=str(adapter),
        max_seq_length=32768,
        load_in_4bit=False,
        full_finetuning=False,
    )
    save = getattr(model, "save_pretrained_merged", None)
    if save is None:
        raise SystemExit("This Unsloth build cannot merge an MLX LoRA.")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        save(str(dest), tokenizer, save_method="merged_16bit")
    except TypeError:
        save(str(dest), tokenizer)


def dwq_command(teacher: Path, dest: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlx_lm.dwq",
        "--model",
        str(teacher),
        "--mlx-path",
        str(dest),
        "--bits",
        "4",
    ]


def run_export_reader(runner=None) -> None:
    """Fuse, save F16, DWQ to 4-bit, score that model, print three rates."""
    from playground.compare_lora import print_scores, repos_present, require_saved_trace
    from playground.subswe import _rescore_file, fetch_subswe_repos, load_subswe_tasks, run_subswe, trace_path

    check_adapter(ADAPTER_DIR)
    tasks = load_subswe_tasks()
    require_saved_trace(F16_MODEL, 1)
    require_saved_trace(LORA_MODEL, 3)
    if not repos_present(tasks):
        fetch_subswe_repos(tasks)
    base = resolve_cached_base()
    run = runner or _run
    BF16_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fusing {base} into {BF16_DIR}", flush=True)
    fuse_bf16(ADAPTER_DIR, BF16_DIR)
    print(f"DWQ 4-bit into {FOUR_BIT_DIR}", flush=True)
    run(dwq_command(BF16_DIR, FOUR_BIT_DIR))
    dest = trace_path(READER_4BIT_MODEL, 1)
    if dest.is_file():
        dest.unlink()
    asyncio.run(run_subswe(settings(), READER_4BIT_MODEL, 1, grade=False))
    task_map = {task["id"]: task for task in tasks}
    print_scores(
        [
            _rescore_file(trace_path(F16_MODEL, 1), task_map, F16_MODEL, 1),
            _rescore_file(trace_path(LORA_MODEL, 3), task_map, LORA_MODEL, 3),
            _rescore_file(dest, task_map, READER_4BIT_MODEL, 1),
        ]
    )


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)
