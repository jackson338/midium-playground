"""Smoke scores. No teacher is selected from these numbers."""

from __future__ import annotations

import re
from pathlib import Path

from playground.harness.tools import TOOL_NAMES

_PATH_RE = re.compile(r"[\w./@+-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|rb|c|h|cc|cpp|md|toml|json|yml|yaml)")
_LINE_RE = re.compile(r"(?:line\s+|:)(\d{1,6})\b", re.IGNORECASE)


def score_episode(episode: dict, work_root: Path) -> dict:
    rounds = episode.get("tool_rounds") or []
    names_ok = 0
    args_ok = 0
    cited_ok = 0
    cited = 0
    for item in rounds:
        name = item.get("name")
        if name in TOOL_NAMES:
            names_ok += 1
        args = item.get("arguments") or {}
        if _args_ok(name, args):
            args_ok += 1
        if name == "read_file":
            path = args.get("path")
            if path:
                cited += 1
                if _page_matches(work_root, str(path), item.get("result") or {}):
                    cited_ok += 1
        if name == "grep":
            for match in (item.get("result") or {}).get("matches") or []:
                rel = match.get("path")
                line = match.get("line")
                if not rel or not isinstance(line, int):
                    continue
                cited += 1
                if _line_exists(work_root, rel, line, match.get("text") or ""):
                    cited_ok += 1
    report = episode.get("report") or ""
    report_hits, report_total = _report_paths(work_root, report, rounds)
    n = max(1, len(rounds))
    max_round = max((int(item.get("round") or 0) for item in rounds), default=-1)
    stopped = max_round < 8
    return {
        "valid_tool_names": names_ok / n if rounds else 1.0,
        "valid_tool_args": args_ok / n if rounds else 1.0,
        "cited_lines_exist": (cited_ok / cited) if cited else 1.0,
        "stopped_by_round_8": 1.0 if stopped else 0.0,
        "report_paths_exist": (report_hits / report_total) if report_total else 1.0,
        "tool_rounds": len(rounds),
        "token_estimate": episode.get("token_estimate") or 0,
        "has_report": bool(report.strip()),
    }


def _args_ok(name: str, args: dict) -> bool:
    if name not in TOOL_NAMES or not isinstance(args, dict):
        return False
    if name == "read_file":
        if not isinstance(args.get("path"), str) or not args.get("path"):
            return False
        for key in ("start_line", "end_line"):
            if key in args and not isinstance(args[key], int):
                return False
        if "start_line" in args and "end_line" in args and args["end_line"] < args["start_line"]:
            return False
        return True
    if name in {"grep", "glob", "web_search", "os_bash", "fetch_url", "match_path"}:
        key = {
            "grep": "pattern",
            "glob": "pattern",
            "web_search": "query",
            "os_bash": "command",
            "fetch_url": "url",
            "match_path": "query",
        }[name]
        return isinstance(args.get(key), str) and bool(args.get(key).strip())
    return True


def _resolve(work_root: Path, path: str) -> Path | None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = work_root / candidate
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    try:
        resolved.relative_to(work_root.resolve())
    except ValueError:
        return None
    return resolved


def _page_matches(work_root: Path, path: str, result: dict) -> bool:
    if result.get("error"):
        return False
    resolved = _resolve(work_root, path) or _resolve(work_root, str(result.get("path") or ""))
    if resolved is None or not resolved.is_file():
        return False
    content = result.get("content") or ""
    start = result.get("start_line") or result.get("end_line")
    if not content or not isinstance(start, int):
        return resolved.is_file()
    try:
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    page = "\n".join(lines[start - 1 : start - 1 + content.count("\n") or 1])
    snippet = content.strip().splitlines()
    if not snippet:
        return True
    return snippet[0] in page or snippet[0] in "\n".join(lines)


def _line_exists(work_root: Path, rel: str, line: int, text: str) -> bool:
    resolved = _resolve(work_root, rel)
    if resolved is None or not resolved.is_file() or line < 1:
        return False
    try:
        lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    if line > len(lines):
        return False
    if not text:
        return True
    return text[:80] in lines[line - 1]


def _report_paths(work_root: Path, report: str, rounds: list[dict]) -> tuple[int, int]:
    seen = set(_PATH_RE.findall(report))
    if not seen:
        return 0, 0
    known: set[str] = set()
    for item in rounds:
        result = item.get("result") or {}
        for key in ("path", "absolute_path"):
            if result.get(key):
                known.add(str(result[key]))
        for match in result.get("matches") or []:
            if match.get("path"):
                known.add(str(match["path"]))
        for entry in result.get("entries") or []:
            if entry.get("path"):
                known.add(str(entry["path"]))
    hits = 0
    for path in seen:
        if any(path in item or item.endswith(path) for item in known):
            hits += 1
            continue
        resolved = _resolve(work_root, path)
        if resolved is not None and resolved.exists():
            hits += 1
    return hits, len(seen)


def summarize(per_episode: list[dict]) -> dict:
    if not per_episode:
        return {"episodes": 0}
    keys = (
        "valid_tool_names",
        "valid_tool_args",
        "cited_lines_exist",
        "stopped_by_round_8",
        "report_paths_exist",
        "token_estimate",
    )
    out: dict = {"episodes": len(per_episode)}
    for key in keys:
        values = [float(row[key]) for row in per_episode]
        out[key] = round(sum(values) / len(values), 4)
    out["with_report"] = sum(1 for row in per_episode if row.get("has_report"))
    return out
