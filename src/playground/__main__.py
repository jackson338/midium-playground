"""MacBook commands.

    uv run python -m playground commissions --repos 200 --per-repo 13 --concurrency 8
    uv run python -m playground smoke
    uv run python -m playground generate --teacher "Qwen3.8 Flash Next" --n 1000 --concurrency 8
    uv run python -m playground subswe --model "Gemma 4 E4B" --run 1
    uv run python -m playground subswe --report
    uv run python -m playground benchmark
    uv run python -m playground benchmark compare e4b-baseline after-lora
    uv run python -m playground dry-run
    uv run python -m playground probe --context 16384
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playground.config import (
    COMMISSIONS_PATH,
    LONG_TRACE_MIN_TOKENS,
    FLASH_TEACHER,
    OPENROUTER_CONCURRENCY,
    SMOKE_REPORT_PATH,
    TEACHERS,
    TRAIN_EPISODES,
    TRAIN_PER_REPO,
    TRAIN_REPOS,
    TRACES_DIR,
    UCE_WORK_ROOT,
    settings,
)
from playground.traces import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(prog="playground")
    sub = parser.add_subparsers(dest="cmd", required=True)

    commissions = sub.add_parser("commissions", help="Clone repos and write reader objectives.")
    commissions.add_argument("--repos", type=int, default=TRAIN_REPOS)
    commissions.add_argument("--per-repo", type=int, default=TRAIN_PER_REPO)
    commissions.add_argument("--concurrency", type=int, default=OPENROUTER_CONCURRENCY)

    smoke = sub.add_parser("smoke", help="Same 2 commissions on all three teachers, Laguna XS 2.1 first.")
    smoke.add_argument("--concurrency", type=int, default=6)

    generate = sub.add_parser("generate", help="Run one chosen teacher. --teacher is required.")
    generate.add_argument("--teacher", required=True)
    generate.add_argument("--n", type=int, default=TRAIN_EPISODES)
    generate.add_argument("--concurrency", type=int, default=2)

    sub.add_parser("dry-run", help="Write a few valid JSONL rows with a scripted reader. No API keys.")

    benchmark = sub.add_parser("benchmark", help="Score local Gemma 4 E4B on the fixed reader suite.")
    benchmark.add_argument("--label", default="e4b-baseline")
    benchmark.add_argument("--no-grade", action="store_true")
    benchmark.add_argument("--work-root", default=str(UCE_WORK_ROOT))
    bench_sub = benchmark.add_subparsers(dest="bench_cmd")
    compare = bench_sub.add_parser("compare", help="Show tasks that moved between two reports.")
    compare.add_argument("before")
    compare.add_argument("after")

    subswe = sub.add_parser("subswe", help="Run or rescore the public SubSWE reader benchmark.")
    subswe.add_argument("--model", default="")
    subswe.add_argument("--run", type=int, default=1)
    subswe.add_argument("--report", action="store_true")
    subswe.add_argument("--no-grade", action="store_true")

    probe = sub.add_parser("probe", help="One-step Gemma 4 E4B LoRA memory probe. Studio only.")
    probe.add_argument("--context", type=int, required=True, choices=(16384, 32768, 98304))

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
    if args.cmd == "benchmark":
        _benchmark(args)
        return
    if args.cmd == "subswe":
        _subswe(args)
        return
    if args.cmd == "probe":
        from playground.probe import run_probe_cli

        run_probe_cli(args.context)
        return


def _subswe(args: argparse.Namespace) -> None:
    from playground.subswe import run_subswe, write_report

    if args.report:
        write_report()
        return
    if not args.model:
        raise SystemExit("Pass --model or --report.")
    asyncio.run(
        run_subswe(settings(), args.model, args.run, grade=not args.no_grade)
    )


def _benchmark(args: argparse.Namespace) -> None:
    from pathlib import Path

    from playground.benchmark import compare_reports, run_benchmark

    if args.bench_cmd == "compare":
        print(compare_reports(args.before, args.after), flush=True)
        return
    asyncio.run(
        run_benchmark(
            settings(),
            label=args.label,
            grade=not args.no_grade,
            work_root=Path(args.work_root),
        )
    )


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

    allowed = (*TEACHERS, FLASH_TEACHER)
    if teacher not in allowed:
        known = ", ".join(allowed)
        raise SystemExit(f"Unknown teacher {teacher!r}. Choose one of: {known}")
    rows = read_jsonl(COMMISSIONS_PATH)
    if not rows:
        raise SystemExit(
            "No commissions yet. Run:\n"
            "  uv run python -m playground commissions --repos 200 --per-repo 13 --concurrency 8"
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
