"""MacBook commands.

    uv run python -m playground commissions --repos 200 --per-repo 25
    uv run python -m playground smoke
    uv run python -m playground generate --teacher "Laguna S 2.1" --n 5000
    uv run python -m playground dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playground.config import (
    COMMISSIONS_PATH,
    LONG_TRACE_MIN_TOKENS,
    SMOKE_REPORT_PATH,
    TEACHERS,
    TRACES_DIR,
    settings,
)
from playground.traces import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="playground")
    sub = parser.add_subparsers(dest="cmd", required=True)

    commissions = sub.add_parser("commissions", help="Clone repos and write reader objectives.")
    commissions.add_argument("--repos", type=int, default=200)
    commissions.add_argument("--per-repo", type=int, default=25)
    commissions.add_argument("--concurrency", type=int, default=16)

    smoke = sub.add_parser("smoke", help="Same 10 commissions on all three teachers.")
    smoke.add_argument("--concurrency", type=int, default=6)

    generate = sub.add_parser("generate", help="Run one chosen teacher. --teacher is required.")
    generate.add_argument("--teacher", required=True)
    generate.add_argument("--n", type=int, default=5000)
    generate.add_argument("--concurrency", type=int, default=8)

    sub.add_parser("dry-run", help="Write a few valid JSONL rows with a scripted reader. No API keys.")

    args = parser.parse_args()
    if args.cmd == "commissions":
        from playground.commissions import write_commissions

        asyncio.run(
            write_commissions(
                settings(),
                repos=args.repos,
                per_repo=args.per_repo,
                concurrency=args.concurrency,
            )
        )
        return
    if args.cmd == "smoke":
        asyncio.run(_smoke(args.concurrency))
        return
    if args.cmd == "generate":
        asyncio.run(_generate(args.teacher, args.n, args.concurrency))
        return
    if args.cmd == "dry-run":
        _dry_run()
        return


async def _smoke(concurrency: int) -> None:
    from playground.commissions import write_smoke_set
    from playground.run import run_teacher

    cfg = settings()
    commissions = await write_smoke_set(cfg)
    print(f"Smoke set: {len(commissions)} commissions", flush=True)
    report = {
        "commissions": [
            {"repo": item["repo"], "commit": item["commit"], "objective": item["objective"]}
            for item in commissions
        ],
        "teachers": {},
        "selected_teacher": None,
        "note": "No teacher was selected. Compare the scores, then pass --teacher to generate.",
    }
    for teacher in TEACHERS:
        print(f"Teacher {teacher}", flush=True)
        summary = await run_teacher(
            cfg,
            teacher,
            commissions,
            concurrency=concurrency,
            dest=TRACES_DIR / "smoke" / f"{teacher.replace(' ', '_').replace('.', '_')}.jsonl",
        )
        report["teachers"][teacher] = summary
        print(json.dumps({teacher: summary}, indent=2), flush=True)
    SMOKE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {SMOKE_REPORT_PATH}", flush=True)


async def _generate(teacher: str, n: int, concurrency: int) -> None:
    from playground.run import run_teacher

    if teacher not in TEACHERS:
        known = ", ".join(TEACHERS)
        raise SystemExit(f"Unknown teacher {teacher!r}. Choose one of: {known}")
    rows = read_jsonl(COMMISSIONS_PATH)
    if not rows:
        raise SystemExit(
            "No commissions yet. Run:\n"
            "  uv run python -m playground commissions --repos 200 --per-repo 25 --concurrency 16"
        )
    from playground.run import teacher_slug

    chosen = rows[:n]
    summary = await run_teacher(settings(), teacher, chosen, concurrency=concurrency)
    traces = read_jsonl(TRACES_DIR / f"{teacher_slug(teacher)}.jsonl")
    long = sum(1 for row in traces if (row.get("token_estimate") or 0) >= LONG_TRACE_MIN_TOKENS)
    summary["episodes_on_disk"] = len(traces)
    summary["between_32k_and_96k"] = long
    print(json.dumps(summary, indent=2), flush=True)


def _dry_run() -> None:
    from playground.harness.loop import run_reader
    from playground.traces import append_jsonl, build_episode

    root = TRACES_DIR.parent / "repos" / "_dry_fixture"
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"def function_{i}():\n    return {i}\n" for i in range(1, 801))
    (src / "widget.py").write_text(body, encoding="utf-8")
    (root / "README.md").write_text("# dry fixture\n", encoding="utf-8")

    async def scripted(messages, tools):
        calls = [
            m
            for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        ]
        if not calls:
            return {
                "role": "assistant",
                "content": "<think>The objective names a line range.</think>",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps(
                                {"path": "src/widget.py", "start_line": 1, "end_line": 80}
                            ),
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "reasoning_content": "The page shows function_1 through the later helpers.",
            "content": "src/widget.py defines function_1 at the top of the file and returns its index.",
        }

    async def go():
        return await run_reader(
            work_root=root,
            objective="Explain src/widget.py lines 1-80. What does function_1 return?",
            complete=scripted,
        )

    result = asyncio.run(go())
    dest = TRACES_DIR / "dry.jsonl"
    if dest.exists():
        dest.unlink()
    for index in (1, 2):
        row = build_episode(
            episode_id=f"dry-{index}",
            repo="local/_dry_fixture",
            commit="dry",
            teacher="scripted",
            commission="Explain src/widget.py lines 1-80. What does function_1 return?",
            loop_result=result,
        )
        if row is None:
            raise SystemExit("Dry-run episode was dropped.")
        append_jsonl(dest, row)
    rows = read_jsonl(dest)
    required = {
        "id",
        "repo",
        "commit",
        "teacher",
        "commission",
        "tool_rounds",
        "thinking",
        "report",
        "token_estimate",
        "char_length",
        "split",
        "raw_turn",
    }
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise SystemExit(f"Dry-run row missing {sorted(missing)}")
        if "<think>" in row["report"]:
            raise SystemExit("Thinking leaked into the report.")
        if not row["thinking"]:
            raise SystemExit("Thinking field was empty.")
        page = row["tool_rounds"][0]["result"]["content"]
        if "def function_1" not in page:
            raise SystemExit("read_file did not return raw lines.")
    print(f"Wrote {len(rows)} episodes to {dest}", flush=True)


if __name__ == "__main__":
    main()
