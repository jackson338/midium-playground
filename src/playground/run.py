"""Run reader commissions through the copied harness on Midium Cloud."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from playground.cloud import CloudError, MidiumCloud, OpenRouter
from playground.config import (
    FLASH_TEACHER,
    MAX_CONTEXT_TOKENS,
    MIDIUM_MAX_CONCURRENCY,
    Settings,
    TRACES_DIR,
)
from playground.harness.loop import run_reader
from playground.score import score_episode, summarize
from playground.traces import append_jsonl, build_episode, existing_ids, read_jsonl


def teacher_slug(teacher: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in teacher).strip("-")


def episode_id(repo: str, commit: str, objective: str, teacher: str) -> str:
    digest = hashlib.sha256(f"{repo}\n{commit}\n{objective}\n{teacher}".encode()).hexdigest()[:16]
    return f"{repo.replace('/', '__')}-{digest}"


async def run_teacher(
    cfg: Settings,
    teacher: str,
    commissions: list[dict],
    *,
    concurrency: int,
    dest: Path | None = None,
) -> dict:
    via_openrouter = teacher == FLASH_TEACHER
    if via_openrouter:
        cfg.require_openrouter()
    else:
        cfg.require_midium()
        if concurrency > MIDIUM_MAX_CONCURRENCY:
            print(f"Midium Cloud concurrency capped at {MIDIUM_MAX_CONCURRENCY}", flush=True)
            concurrency = MIDIUM_MAX_CONCURRENCY
    path = dest or (TRACES_DIR / f"{teacher_slug(teacher)}.jsonl")
    done = existing_ids(path)
    pending = []
    for item in commissions:
        eid = episode_id(item["repo"], item["commit"], item["objective"], teacher)
        if eid not in done:
            pending.append((eid, item))
    client = OpenRouter(cfg, timeout=180.0) if via_openrouter else MidiumCloud(cfg, teacher)
    sem = asyncio.Semaphore(concurrency)
    written = 0
    dropped = 0
    failed = 0

    async def one(eid: str, item: dict) -> None:
        nonlocal written, dropped, failed
        async with sem:
            result = None
            for attempt in range(1, 4):
                try:
                    result = await run_reader(
                        work_root=Path(item["path"]),
                        objective=item["objective"],
                        complete=client.complete,
                    )
                    break
                except CloudError as exc:
                    if exc.status != 500 or attempt == 3:
                        failed += 1
                        print(f"  fail {item['repo']}: {exc}", flush=True)
                        return
                    print(f"  retry episode {attempt} {item['repo']}: {exc}", flush=True)
                    await asyncio.sleep(15 * attempt)
                except OSError as exc:
                    failed += 1
                    print(f"  fail {item['repo']}: {exc}", flush=True)
                    return
            if result is None:
                return
            row = build_episode(
                episode_id=eid,
                repo=item["repo"],
                commit=item["commit"],
                teacher=teacher,
                commission=item["objective"],
                loop_result=result,
            )
            if row is None:
                dropped += 1
                print(f"  drop {eid} over {MAX_CONTEXT_TOKENS} tokens", flush=True)
                return
            append_jsonl(path, row)
            written += 1
            print(
                f"  {written} {item['repo']} tokens≈{row['token_estimate']} rounds={len(row['tool_rounds'])}",
                flush=True,
            )

    try:
        await asyncio.gather(*(one(eid, item) for eid, item in pending))
    finally:
        await client.aclose()
    roots = {
        episode_id(item["repo"], item["commit"], item["objective"], teacher): Path(item["path"])
        for item in commissions
    }
    summary = summarize(score_trace(path, roots))
    summary.update({"written": written, "dropped_over_context": dropped, "failed": failed, "path": str(path)})
    return summary


def score_trace(path: Path, roots: dict[str, Path]) -> list[dict]:
    """Score saved episodes for this run's commissions, including rows written earlier."""
    if not path.is_file():
        return []
    scored = []
    for row in read_jsonl(path):
        root = roots.get(row.get("id") or "")
        if root is None or not Path(root).is_dir():
            continue
        scored.append(score_episode(row, Path(root)))
    return scored
