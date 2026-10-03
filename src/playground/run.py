"""Run reader commissions through the copied harness on Midium Cloud."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from playground.cloud import CloudError, MidiumCloud
from playground.config import MAX_CONTEXT_TOKENS, Settings, TRACES_DIR
from playground.harness.loop import run_reader
from playground.score import score_episode, summarize
from playground.traces import append_jsonl, build_episode, existing_ids


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
    cfg.require_midium()
    path = dest or (TRACES_DIR / f"{teacher_slug(teacher)}.jsonl")
    done = existing_ids(path)
    pending = []
    for item in commissions:
        eid = episode_id(item["repo"], item["commit"], item["objective"], teacher)
        if eid not in done:
            pending.append((eid, item))
    cloud = MidiumCloud(cfg, teacher)
    sem = asyncio.Semaphore(concurrency)
    written = 0
    dropped = 0
    failed = 0
    scored: list[dict] = []

    async def one(eid: str, item: dict) -> None:
        nonlocal written, dropped, failed
        async with sem:
            try:
                result = await run_reader(
                    work_root=Path(item["path"]),
                    objective=item["objective"],
                    complete=cloud.complete,
                )
            except (CloudError, OSError) as exc:
                failed += 1
                print(f"  fail {item['repo']}: {exc}", flush=True)
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
            scored.append(score_episode(row, Path(item["path"])))
            print(
                f"  {written} {item['repo']} tokens≈{row['token_estimate']} rounds={len(row['tool_rounds'])}",
                flush=True,
            )

    try:
        await asyncio.gather(*(one(eid, item) for eid, item in pending))
    finally:
        await cloud.aclose()
    summary = summarize(scored)
    summary.update({"written": written, "dropped_over_context": dropped, "failed": failed, "path": str(path)})
    return summary
