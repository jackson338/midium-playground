"""JSONL episodes. One row per reader commission.

Fields the later LoRA reads: id, repo, commit, teacher, commission, tool
rounds, thinking, report, token estimate, char length, split. The raw
Midium turn is stored on the same row and is not required on the Studio.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from playground.config import MAX_CONTEXT_TOKENS


def estimate_tokens(text: str) -> int:
    """Rough token count. Four characters per token is enough to drop overflows."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def split_for(episode_id: str) -> str:
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
    return "holdout" if int(digest[:2], 16) < 26 else "train"  # ~10%


def build_episode(
    *,
    episode_id: str,
    repo: str,
    commit: str,
    teacher: str,
    commission: str,
    loop_result: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a row, or None when the transcript is over the 96k cap."""
    tool_rounds = []
    for item in loop_result.get("tool_rounds") or []:
        tool_rounds.append(
            {
                "round": item.get("round"),
                "name": item.get("name"),
                "arguments": item.get("arguments"),
                "result": item.get("result"),
            }
        )
    row = {
        "id": episode_id,
        "repo": repo,
        "commit": commit,
        "teacher": teacher,
        "commission": commission,
        "tool_rounds": tool_rounds,
        "thinking": loop_result.get("thinking") or [],
        "report": loop_result.get("report") or "",
        "stop_reason": loop_result.get("stop_reason"),
        "raw_turn": {
            "messages": loop_result.get("messages") or [],
            "raw_steps": _strip_raw(loop_result.get("raw_steps") or []),
        },
    }
    blob = json.dumps(row, ensure_ascii=False)
    row["char_length"] = len(blob)
    row["token_estimate"] = estimate_tokens(blob)
    row["split"] = split_for(episode_id)
    if row["token_estimate"] > MAX_CONTEXT_TOKENS:
        return None
    return row


def _strip_raw(steps: list[dict]) -> list[dict]:
    cleaned = []
    for step in steps:
        assistant = dict(step.get("assistant") or {})
        assistant.pop("_raw", None)
        cleaned.append({**step, "assistant": assistant})
    return cleaned


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def existing_ids(path: Path) -> set[str]:
    return {str(row.get("id")) for row in read_jsonl(path) if row.get("id")}
