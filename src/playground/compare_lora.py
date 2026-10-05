"""Score base F16 Gemma, train 100 tool-call examples, score that adapter.

No judge. The two pass rates are printed separately.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from playground.config import ROOT, settings
from playground.subswe import (
    F16_MODEL,
    LORA_MODEL,
    fetch_subswe_repos,
    load_subswe_tasks,
    run_subswe,
    trace_path,
)
from playground.train import NEXT_CHECKPOINT_DIR, NEXT_SKIP, TOOLS_CHECKPOINT_DIR, run_train_cli


def repos_present(tasks: list[dict] | None = None) -> bool:
    for task in tasks or load_subswe_tasks():
        path = Path(task["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_dir():
            return False
    return True


def drop_previous_lora_trace(path: Path | None = None) -> None:
    """The old 1/40 file must not be resumed."""
    dest = path or trace_path(LORA_MODEL, 1)
    if dest.is_file():
        dest.unlink()


def print_scores(rows: list[dict]) -> None:
    for row in rows:
        print(
            f"{row['model']}: pass_rate={row['pass_rate']} "
            f"passed={row['passed']}/{row['attempted']} calls={row['call_count']} run={row.get('run')}",
            flush=True,
        )
        for kind, stats in row["by_kind"].items():
            print(f"  {kind} {stats['passed']}/{stats['attempted']}", flush=True)
        for item in row["tasks"]:
            flags = item.get("deterministic") or {}
            if item.get("error") or flags.get("invalid"):
                print(f"  invalid {item['id']}", flush=True)
                continue
            if flags.get("passed"):
                continue
            failed = [name for name, value in flags.items() if value is False and name != "passed"]
            print(f"  fail {item['id']}: {', '.join(failed)}", flush=True)


def print_pair(base: dict, adapted: dict) -> None:
    print_scores([base, adapted])


def run_compare_lora() -> None:
    tasks = load_subswe_tasks()
    if not repos_present(tasks):
        fetch_subswe_repos(tasks)
    cfg = settings()
    asyncio.run(run_subswe(cfg, F16_MODEL, 1, grade=False))
    run_train_cli(32768, 1, examples=100, checkpoint=TOOLS_CHECKPOINT_DIR)
    drop_previous_lora_trace()
    asyncio.run(run_subswe(cfg, LORA_MODEL, 1, grade=False, adapter=TOOLS_CHECKPOINT_DIR))
    from playground.subswe import _rescore_file

    task_map = {task["id"]: task for task in tasks}
    base = _rescore_file(trace_path(F16_MODEL, 1), task_map, F16_MODEL, 1)
    adapted = _rescore_file(trace_path(LORA_MODEL, 1), task_map, LORA_MODEL, 1)
    print_pair(base, adapted)


def require_saved_trace(model: str, run: int) -> Path:
    """train-next prints these files. It does not rerun them."""
    path = trace_path(model, run)
    if not path.is_file():
        raise SystemExit(f"Missing saved score {path}. train-next does not rerun it.")
    return path


def drop_lora_run2(path: Path | None = None) -> None:
    """A fresh score of the continued adapter. Run 1 stays."""
    dest = path or trace_path(LORA_MODEL, 2)
    if dest.is_file():
        dest.unlink()


def run_train_next() -> None:
    """Continue the 100-step adapter on the next 100 episodes, then score it."""
    if not (TOOLS_CHECKPOINT_DIR / "adapter_config.json").is_file():
        raise SystemExit(f"100-step adapter not found: {TOOLS_CHECKPOINT_DIR}")
    tasks = load_subswe_tasks()
    require_saved_trace(F16_MODEL, 1)
    require_saved_trace(LORA_MODEL, 1)
    if not repos_present(tasks):
        fetch_subswe_repos(tasks)
    run_train_cli(
        32768,
        1,
        examples=NEXT_SKIP,
        skip=NEXT_SKIP,
        checkpoint=NEXT_CHECKPOINT_DIR,
        resume=TOOLS_CHECKPOINT_DIR,
    )
    drop_lora_run2()
    cfg = settings()
    asyncio.run(run_subswe(cfg, LORA_MODEL, 2, grade=False, adapter=NEXT_CHECKPOINT_DIR))
    from playground.subswe import _rescore_file

    task_map = {task["id"]: task for task in tasks}
    print_scores(
        [
            _rescore_file(trace_path(F16_MODEL, 1), task_map, F16_MODEL, 1),
            _rescore_file(trace_path(LORA_MODEL, 1), task_map, LORA_MODEL, 1),
            _rescore_file(trace_path(LORA_MODEL, 2), task_map, LORA_MODEL, 2),
        ]
    )
