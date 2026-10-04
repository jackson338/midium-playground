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
from playground.train import TOOLS_CHECKPOINT_DIR, run_train_cli


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


def print_pair(base: dict, adapted: dict) -> None:
    for row in (base, adapted):
        print(
            f"{row['model']}: pass_rate={row['pass_rate']} "
            f"passed={row['passed']}/{row['attempted']} calls={row['call_count']}",
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
