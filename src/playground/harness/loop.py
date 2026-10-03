"""Research junior loop.

Mirrors the UCE researcher: the final assistant message is the report,
there is no submit tool, a wrap-up nudge lands at iteration 6, and the
hard cap is 8 (``RESEARCH_WRAPUP_ITER`` / ``_RESEARCH_ITERS``). Thinking
stays in its own field.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Awaitable, Callable

from playground.config import RESEARCH_MAX_ITERS, RESEARCH_WRAPUP_ITER
from playground.harness.shell import Workspace
from playground.harness.skill import ITER_CAP_WRAPUP, RESEARCH_WRAPUP, format_research_skill
from playground.harness.tools import ReaderTools, openai_tools

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

Complete = Callable[[list[dict], list[dict] | None], Awaitable[dict]]


def split_thinking(content: str) -> tuple[str, str]:
    """Pull ``<think>`` blocks out of assistant text. The report is the rest."""
    if not content:
        return "", ""
    thoughts = [chunk.strip() for chunk in _THINK_RE.findall(content) if chunk.strip()]
    report = _THINK_RE.sub("", content).strip()
    return "\n\n".join(thoughts), report


async def run_reader(
    *,
    work_root: Path,
    objective: str,
    complete: Complete,
) -> dict[str, Any]:
    """Run one commission. ``complete`` talks to Midium Cloud (or a fake).

    ``complete(messages, tools)`` returns an OpenAI-style assistant message
    dict. ``tools`` is None on the final no-tools wrap-up call.
    """
    shell = Workspace(work_root)
    tools = ReaderTools(shell)
    messages: list[dict] = [
        {"role": "system", "content": format_research_skill(str(work_root))},
        {"role": "user", "content": objective},
    ]
    schemas = openai_tools()
    tool_rounds: list[dict] = []
    thinking_steps: list[dict] = []
    raw_steps: list[dict] = []
    nudged = False
    report = ""
    stop_reason = "report"

    for iteration in range(RESEARCH_MAX_ITERS):
        if iteration >= RESEARCH_WRAPUP_ITER and not nudged:
            messages.append({"role": "system", "content": RESEARCH_WRAPUP})
            nudged = True
        message = await complete(messages, schemas)
        raw_steps.append({"iteration": iteration, "assistant": message})
        thinking, visible = _thinking_from_message(message)
        if thinking:
            thinking_steps.append({"round": iteration, "text": thinking})
        tool_calls = message.get("tool_calls") or []
        assistant = {
            "role": "assistant",
            "content": visible or message.get("content") or "",
        }
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        if not tool_calls:
            report = visible
            stop_reason = "report"
            break
        for call in tool_calls:
            name, args, call_id = _parse_call(call)
            result = tools.dispatch(name, args)
            tool_rounds.append(
                {
                    "round": iteration,
                    "name": name,
                    "arguments": args,
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )
    else:
        messages.append({"role": "system", "content": ITER_CAP_WRAPUP})
        message = await complete(messages, None)
        raw_steps.append({"iteration": RESEARCH_MAX_ITERS, "assistant": message, "tools": None})
        thinking, visible = _thinking_from_message(message)
        if thinking:
            thinking_steps.append({"round": RESEARCH_MAX_ITERS, "text": thinking})
        report = visible
        stop_reason = "max_tool_iterations"

    return {
        "report": report,
        "thinking": thinking_steps,
        "tool_rounds": tool_rounds,
        "stop_reason": stop_reason,
        "messages": messages,
        "raw_steps": raw_steps,
    }


def _thinking_from_message(message: dict) -> tuple[str, str]:
    pieces: list[str] = []
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip())
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = ""
    tagged, visible = split_thinking(content)
    if tagged:
        pieces.append(tagged)
    return "\n\n".join(pieces), visible


def _parse_call(call: dict) -> tuple[str, dict, str]:
    fn = call.get("function") or {}
    name = fn.get("name") or call.get("name") or ""
    raw = fn.get("arguments")
    if raw is None:
        raw = call.get("arguments") or {}
    if isinstance(raw, str):
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw.strip() or "{}")
            args = parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            args = {}
    elif isinstance(raw, dict):
        args = raw
    else:
        args = {}
    call_id = str(call.get("id") or f"call_{name}")
    return name, args, call_id
