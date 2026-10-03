"""Fixed reader benchmark for Gemma 4 E4B.

The same tasks are scored before and after the LoRA. Deterministic checks
look at tool calls and the report. Gemma 4 26B A4B on Midium Cloud adds a
good/bad grade and does not override a failed check.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from playground.cloud import CloudError, MidiumCloud, parse_json_object
from playground.config import (
    BENCHMARK_TASKS_PATH,
    BENCHMARKS_DIR,
    GRADER_MODEL,
    STUDENT_MODEL,
    Settings,
    UCE_WORK_ROOT,
)
from playground.harness.loop import run_reader
from playground.harness.tools import TOOL_NAMES
from playground.score import _args_ok, calls_within_cap

_ABSENCE = re.compile(
    r"("
    r"cannot locate|could not find|couldn't find|does not exist|doesn't exist|"
    r"not present|not found|no matches|no such function|unable to find|"
    r"not in (this|the) (repo|codebase|files)|"
    r"i (cannot|can't|could not) find"
    r")",
    re.IGNORECASE,
)

_GRADER_SYSTEM = """\
You grade a code-reader's final report. Return JSON only, no markdown:
{"grade":"good"|"bad","reason":"one sentence"}

grade is good only when the report states the facts in the gold note and does
not contradict them. grade is bad when a required fact is missing, wrong, or
the report claims the target is absent even though the gold note says it exists.
"""


def load_tasks(path: Path | None = None) -> list[dict]:
    raw = json.loads((path or BENCHMARK_TASKS_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise SystemExit(f"No tasks in {path or BENCHMARK_TASKS_PATH}")
    return raw


def repo_commit(work_root: Path) -> str:
    proc = subprocess.run(
        ["git", "-C", str(work_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def check_task(task: dict, loop_result: dict, work_root: Path) -> dict:
    """Deterministic flags. A missing target file makes the task invalid."""
    checks = task.get("checks") or {}
    rel_path = (checks.get("path") or "").strip()
    symbol = (checks.get("symbol") or "").strip()
    quote = checks.get("quote") or ""
    must_read = bool(checks.get("must_read"))
    expect_absent = bool(checks.get("expect_absent"))
    forbid_absence = bool(checks.get("forbid_absence_claim"))
    report = loop_result.get("report") or ""
    rounds = loop_result.get("tool_rounds") or []

    target = (work_root / rel_path) if rel_path else None
    invalid = bool(rel_path) and (target is None or not target.is_file())
    if expect_absent and symbol and _symbol_defined(work_root, symbol):
        invalid = True

    valid_tools = _valid_tools(rounds)
    stopped = calls_within_cap(rounds, work_root)
    found_path = _found_path(rounds, work_root, rel_path) if rel_path else True
    named_path = _report_names_path(report, rel_path) if rel_path else True
    read_ok, page_covers = _read_status(rounds, work_root, rel_path, quote if must_read else "")
    report_has_quote = _has_quote(report, quote) if quote and not expect_absent else True
    symbol_seen = (not symbol) or expect_absent or symbol in report
    false_absence = bool(forbid_absence and _ABSENCE.search(report))
    claimed_absent = bool(_ABSENCE.search(report)) if expect_absent else True

    flags = {
        "invalid": invalid,
        "valid_tools": valid_tools,
        "stopped_by_round_8": stopped,
        "found_path": found_path,
        "named_path": named_path,
        "read_file": read_ok if must_read else True,
        "page_covers_quote": page_covers if must_read and quote else True,
        "report_has_quote": report_has_quote,
        "symbol_seen": symbol_seen,
        "no_false_absence": not false_absence,
        "claimed_absent": claimed_absent,
    }
    if invalid:
        flags["passed"] = False
    else:
        flags["passed"] = all(
            flags[key]
            for key in (
                "valid_tools",
                "stopped_by_round_8",
                "found_path",
                "named_path",
                "read_file",
                "page_covers_quote",
                "report_has_quote",
                "symbol_seen",
                "no_false_absence",
                "claimed_absent",
            )
        )
    return flags


def report_path(label: str) -> Path:
    safe = "".join(ch.lower() if ch.isalnum() or ch in "-_" else "-" for ch in label).strip("-")
    return BENCHMARKS_DIR / f"{safe or 'benchmark'}.json"


async def run_benchmark(
    cfg: Settings,
    *,
    label: str,
    grade: bool,
    work_root: Path | None = None,
) -> dict:
    cfg.require_local()
    if grade:
        cfg.require_midium()
    root = (work_root or UCE_WORK_ROOT).resolve()
    if not root.is_dir():
        raise SystemExit(f"Benchmark work root does not exist: {root}")
    tasks = load_tasks()
    student = MidiumCloud(
        cfg,
        STUDENT_MODEL,
        api_key=cfg.local_api_key,
        base_url=cfg.local_base_url,
    )
    grader = (
        MidiumCloud(cfg, GRADER_MODEL, timeout=120.0)
        if grade
        else None
    )
    rows: list[dict] = []
    try:
        for index, task in enumerate(tasks, 1):
            print(f"  {index}/{len(tasks)} {task['id']}", flush=True)
            rows.append(await _one(student, grader, task, root))
    finally:
        await student.aclose()
        if grader is not None:
            await grader.aclose()
    payload = _summarize(label, root, rows, graded=grade)
    path = report_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    print(
        f"pass_rate={payload['pass_rate']} good_rate={payload['good_rate']}",
        flush=True,
    )
    return payload


def compare_reports(before_label: str, after_label: str) -> str:
    before = json.loads(report_path(before_label).read_text(encoding="utf-8"))
    after = json.loads(report_path(after_label).read_text(encoding="utf-8"))
    before_tasks = {row["id"]: row for row in before.get("tasks") or []}
    lines = [
        f"{before_label} pass_rate={before.get('pass_rate')} good_rate={before.get('good_rate')}",
        f"{after_label} pass_rate={after.get('pass_rate')} good_rate={after.get('good_rate')}",
    ]
    for row in after.get("tasks") or []:
        prior = before_tasks.get(row["id"])
        if prior is None:
            lines.append(f"  {row['id']}: new")
            continue
        det_before = bool((prior.get("deterministic") or {}).get("passed"))
        det_after = bool((row.get("deterministic") or {}).get("passed"))
        grade_before = (prior.get("semantic") or {}).get("grade")
        grade_after = (row.get("semantic") or {}).get("grade")
        if det_before == det_after and grade_before == grade_after:
            continue
        lines.append(
            f"  {row['id']}: check {det_before} -> {det_after}, grade {grade_before} -> {grade_after}"
        )
    if len(lines) == 2:
        lines.append("  no task changed")
    return "\n".join(lines)


async def _one(student: MidiumCloud, grader: MidiumCloud | None, task: dict, root: Path) -> dict:
    try:
        result = await run_reader(
            work_root=root,
            objective=task["objective"],
            complete=student.complete,
        )
    except (CloudError, OSError) as exc:
        return {
            "id": task["id"],
            "kind": task.get("kind"),
            "deterministic": {"passed": False, "invalid": False, "error": str(exc)},
            "semantic": None,
            "report": "",
            "error": str(exc),
        }
    flags = check_task(task, result, root)
    semantic = None
    if grader is not None and not flags.get("invalid"):
        semantic = await _grade(grader, task, result.get("report") or "")
    report = result.get("report") or ""
    return {
        "id": task["id"],
        "kind": task.get("kind"),
        "deterministic": flags,
        "semantic": semantic,
        "report": report if len(report) <= 2000 else report[:2000] + "…",
        "tool_calls": [
            {"name": item.get("name"), "arguments": item.get("arguments")}
            for item in result.get("tool_rounds") or []
        ],
        "error": None,
    }


async def _grade(grader: MidiumCloud, task: dict, report: str) -> dict:
    user = (
        f"Objective:\n{task.get('objective')}\n\n"
        f"Gold note:\n{task.get('gold')}\n\n"
        f"Report:\n{report or '(empty)'}"
    )
    try:
        message = await grader.complete(
            [
                {"role": "system", "content": _GRADER_SYSTEM},
                {"role": "user", "content": user},
            ],
            None,
        )
    except CloudError as exc:
        return {"grade": None, "reason": str(exc)}
    content = message.get("content") or ""
    try:
        parsed = parse_json_object(content)
    except CloudError as exc:
        return {"grade": None, "reason": str(exc), "raw": content[:500]}
    grade = str(parsed.get("grade") or "").strip().lower()
    if grade not in {"good", "bad"}:
        grade = None
    return {"grade": grade, "reason": str(parsed.get("reason") or "").strip()}


def _summarize(label: str, root: Path, rows: list[dict], *, graded: bool) -> dict:
    scored = [row for row in rows if not (row.get("deterministic") or {}).get("invalid")]
    passed = [row for row in scored if (row.get("deterministic") or {}).get("passed")]
    goods = [
        row
        for row in scored
        if (row.get("semantic") or {}).get("grade") == "good"
    ]
    graded_rows = [row for row in scored if (row.get("semantic") or {}).get("grade") in {"good", "bad"}]
    return {
        "label": label,
        "model": STUDENT_MODEL,
        "grader": GRADER_MODEL if graded else None,
        "work_root": str(root),
        "commit": repo_commit(root),
        "tasks": rows,
        "pass_rate": round(len(passed) / len(scored), 4) if scored else 0.0,
        "good_rate": round(len(goods) / len(graded_rows), 4) if graded_rows else None,
    }


def _valid_tools(rounds: list[dict]) -> bool:
    for item in rounds:
        name = item.get("name")
        if name not in TOOL_NAMES or not _args_ok(name, item.get("arguments") or {}):
            return False
    return True


def _found_path(rounds: list[dict], work_root: Path, rel_path: str) -> bool:
    want = rel_path.replace("\\", "/")
    for item in rounds:
        if item.get("name") not in {"grep", "read_file", "glob"}:
            continue
        result = item.get("result") or {}
        if result.get("error"):
            continue
        candidates = []
        args = item.get("arguments") or {}
        if args.get("path"):
            candidates.append(str(args["path"]))
        if result.get("path"):
            candidates.append(str(result["path"]))
        for match in result.get("matches") or []:
            if match.get("path"):
                candidates.append(str(match["path"]))
            if match.get("absolute_path"):
                candidates.append(str(match["absolute_path"]))
        for raw in candidates:
            rel = _as_rel(work_root, raw)
            if rel == want or rel.endswith("/" + want) or want.endswith(rel):
                return True
    return False


def _read_status(
    rounds: list[dict], work_root: Path, rel_path: str, quote: str
) -> tuple[bool, bool]:
    read = False
    covers = not bool(quote)
    want = rel_path.replace("\\", "/")
    for item in rounds:
        if item.get("name") != "read_file":
            continue
        result = item.get("result") or {}
        if result.get("error"):
            continue
        raw = (item.get("arguments") or {}).get("path") or result.get("path") or ""
        rel = _as_rel(work_root, str(raw))
        if rel != want and not rel.endswith("/" + want):
            continue
        read = True
        if quote and quote in (result.get("content") or ""):
            covers = True
    return read, covers


def _report_names_path(report: str, rel_path: str) -> bool:
    if not rel_path:
        return True
    folded = report.replace("\\", "/")
    parts = rel_path.split("/")
    suffix = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return rel_path in folded or suffix in folded


def _has_quote(report: str, quote: str) -> bool:
    if len(quote) <= 3:
        return re.search(rf"\b{re.escape(quote)}\b", report) is not None
    return quote in report


def _symbol_defined(work_root: Path, symbol: str) -> bool:
    for path in work_root.rglob("*.py"):
        if any(part.startswith(".") or part in {"node_modules", ".venv"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(rf"\b{re.escape(symbol)}\b", text):
            return True
    return False


def _as_rel(work_root: Path, raw: str) -> str:
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        path = work_root / path
    try:
        resolved = path.resolve()
        return resolved.relative_to(work_root.resolve()).as_posix()
    except (OSError, ValueError):
        return raw.replace("\\", "/").lstrip("./")
