"""Ask Qwen3.8 Flash on OpenRouter for reader objectives.

Flash writes objectives only. It does not run the reader. Calls run
concurrently across repos.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from playground.cloud import CloudError, OpenRouter, parse_json_object
from playground.config import COMMISSIONS_PATH, SMOKE_COMMISSIONS, SMOKE_COMMISSIONS_PATH, Settings
from playground.repos import catalog, clone_all, discover
from playground.traces import append_jsonl, read_jsonl

_SYSTEM = """\
You write reader commissions for a coding agent that can grep, glob, and read \
raw file pages. You do not answer the commissions and you do not explore \
beyond the file list you are given.

Each commission is ONE slice of ONE repo: find a symbol, trace a call, or \
explain a single region. Never a whole-repo audit. Never a checklist of many \
modules.

Return JSON only, no markdown:
{"objectives":[{"objective":"...","path":"relative/file.py","start_line":1,"end_line":400}]}

Rules:
- `path` must be one of the files in the catalog.
- `start_line` and `end_line` are 1-indexed and inside that file.
- The objective text must name the path and the line range, and ask for one \
specific question (what a function does, who calls it, how one branch works).
- Prefer the longest files. At least 60% of objectives must span 1200 lines \
or more when the catalog has a file that long. Otherwise use the longest \
span the file allows, at least 400 lines when possible.
- Objectives must differ: different regions or different questions.
"""


async def write_commissions(
    cfg: Settings,
    *,
    repos: int,
    per_repo: int,
    concurrency: int,
    dest: Path = COMMISSIONS_PATH,
) -> int:
    cfg.require_openrouter()
    per_repo = max(1, min(per_repo, 40))
    print(f"Searching GitHub for {repos} permissive repos…", flush=True)
    discovered = await discover(cfg, repos)
    print(f"Cloning {len(discovered)} repos…", flush=True)
    cloned = await clone_all(discovered, concurrency=min(8, concurrency))
    print(f"Cloned {len(cloned)}. Writing {per_repo} commissions each…", flush=True)
    already = {row.get("repo") for row in read_jsonl(dest)}
    pending = [repo for repo in cloned if repo["full_name"] not in already]
    client = OpenRouter(cfg)
    sem = asyncio.Semaphore(concurrency)
    written = 0

    async def one(repo: dict) -> int:
        async with sem:
            try:
                rows = await _objectives_for(client, repo, per_repo)
            except CloudError as exc:
                print(f"  skip {repo['full_name']}: {exc}", flush=True)
                return 0
            for row in rows:
                append_jsonl(dest, row)
            print(f"  {repo['full_name']}: {len(rows)}", flush=True)
            return len(rows)

    try:
        counts = await asyncio.gather(*(one(repo) for repo in pending))
    finally:
        await client.aclose()
    written = sum(counts)
    print(f"Wrote {written} commissions to {dest}", flush=True)
    return written


async def write_smoke_set(cfg: Settings, concurrency: int = 4) -> list[dict]:
    """Two commissions, reused for every smoke teacher."""
    existing = read_jsonl(SMOKE_COMMISSIONS_PATH)
    if len(existing) >= SMOKE_COMMISSIONS:
        kept = existing[:SMOKE_COMMISSIONS]
        if len(existing) > SMOKE_COMMISSIONS:
            _write_smoke_rows(kept)
        return kept
    cfg.require_openrouter()
    discovered = await discover(cfg, 8)
    cloned = await clone_all(discovered, concurrency=4)
    ranked = sorted(cloned, key=lambda repo: catalog(Path(repo["path"])).get("long_enough", 0), reverse=True)
    client = OpenRouter(cfg)
    rows: list[dict] = list(existing)
    try:
        for repo in ranked:
            if len(rows) >= SMOKE_COMMISSIONS:
                break
            try:
                batch = await _objectives_for(client, repo, SMOKE_COMMISSIONS - len(rows))
            except CloudError as exc:
                print(f"  skip {repo['full_name']}: {exc}", flush=True)
                continue
            rows.extend(batch)
    finally:
        await client.aclose()
    if len(rows) < SMOKE_COMMISSIONS:
        raise SystemExit(
            f"Only gathered {len(rows)} smoke commissions; need {SMOKE_COMMISSIONS}."
        )
    kept = rows[:SMOKE_COMMISSIONS]
    _write_smoke_rows(kept)
    return kept


def _write_smoke_rows(rows: list[dict]) -> None:
    SMOKE_COMMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_COMMISSIONS_PATH.write_text("", encoding="utf-8")
    for row in rows:
        append_jsonl(SMOKE_COMMISSIONS_PATH, row)


async def _objectives_for(client: OpenRouter, repo: dict, count: int) -> list[dict]:
    info = catalog(Path(repo["path"]))
    if not info["longest"]:
        return []
    user = json.dumps(
        {
            "repo": repo["full_name"],
            "commit": repo["commit"],
            "count": count,
            "files": info["longest"][:25],
        },
        indent=2,
    )
    known = {item["path"]: item for item in info["longest"]}
    text = await client.complete_text(_SYSTEM, user + f"\n\nWrite exactly {count} objectives.")
    rows = _rows_from_model(text, repo, count, known)
    if len(rows) < count:
        reminder = (
            f"\n\nThe last reply produced {len(rows)} usable objectives. "
            f"Write exactly {count}. Every path must be copied from the catalog."
        )
        text = await client.complete_text(_SYSTEM, user + reminder)
        rows = _rows_from_model(text, repo, count, known)
    return rows


def _rows_from_model(text: str, repo: dict, count: int, known: dict) -> list[dict]:
    parsed = parse_json_object(text)
    objectives = parsed.get("objectives") or []
    rows = []
    seen: set[str] = set()
    for item in objectives:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        objective = str(item.get("objective") or "").strip()
        if path not in known or not objective or objective in seen:
            continue
        file_lines = int(known[path]["lines"])
        try:
            start = max(1, int(item.get("start_line") or 1))
            end = int(item.get("end_line") or min(file_lines, start + 1199))
        except (TypeError, ValueError):
            continue
        end = min(file_lines, max(start, end))
        seen.add(objective)
        rows.append(
            {
                "repo": repo["full_name"],
                "commit": repo["commit"],
                "path": repo["path"],
                "license": repo.get("license"),
                "focus_path": path,
                "start_line": start,
                "end_line": end,
                "objective": objective,
            }
        )
        if len(rows) >= count:
            break
    return rows
