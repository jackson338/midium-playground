"""Gemma 4 tool-call turns.

Training renders these messages with the model's own chat template.
The local runner parses one tool-call span and ignores anything after it,
including a tool result the model wrote itself.
"""

from __future__ import annotations

import json
import re

from playground.harness.loop import split_thinking
from playground.harness.skill import format_research_skill
from playground.harness.tools import TOOL_NAMES
from playground.probe import _read_pages

_CALL_OPEN = "<|tool_call>call:"
_CALL_CLOSE = "<tool_call|>"
_RESPONSE_OPEN = "<|tool_response>"
_STRING = '<|"|>'
TOOL_TRAIN_EXAMPLES = 100


def episode_messages(episode: dict, page_texts: list[str], skill: str | None = None) -> list[dict]:
    """System skill, user commission, real tool_calls, tool results, then the report."""
    pages = list(page_texts)
    messages: list[dict] = [
        {"role": "system", "content": skill if skill is not None else format_research_skill(episode.get("repo") or "")},
        {"role": "user", "content": episode.get("commission") or ""},
    ]
    cursor = 0
    for index, item in enumerate(episode.get("tool_rounds") or []):
        name = item.get("name") or ""
        arguments = dict(item.get("arguments") or {})
        call_id = f"call_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        )
        result = dict(item.get("result") or {})
        if name == "read_file":
            result["content"] = pages[cursor] if cursor < len(pages) else ""
            cursor += 1
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )
    messages.append({"role": "assistant", "content": episode.get("report") or ""})
    return messages


def pack_tool_example(episode: dict, limit: int, count_tokens, render) -> dict | None:
    """Trim read_file pages until the rendered template fits. Never pad."""
    pages = _read_pages(episode)
    kept = list(pages)

    def rendered(page_texts: list[str]) -> str:
        return render(episode_messages(episode, page_texts))

    while kept and count_tokens(rendered(kept)) > limit:
        for index in range(len(kept) - 1, -1, -1):
            if kept[index]:
                kept[index] = ""
                break
        else:
            break
    text = rendered(kept)
    tokens = count_tokens(text)
    if tokens > limit:
        return None
    return {"text": text, "tokens": tokens, "pages_used": sum(1 for page in kept if page)}


def build_tool_train_set(rows: list[dict], limit: int, count_tokens, render, n: int = TOOL_TRAIN_EXAMPLES):
    """First ``n`` train rows that fit, in file order. Holdout rows are skipped."""
    packed: list[dict] = []
    skipped = 0
    for row in rows:
        if len(packed) >= n:
            break
        if row.get("split") != "train":
            skipped += 1
            continue
        item = pack_tool_example(row, limit, count_tokens, render)
        if item is None or item["tokens"] > limit:
            skipped += 1
            continue
        packed.append(item)
    if not packed:
        raise SystemExit("No train episodes fit under the 32k cap.")
    return packed, skipped


def require_tool_call_loss(text: str) -> None:
    """Stop unless this example is a Gemma tool-call string."""
    if "<|tool_call>call:" not in text or '{"name"' in text:
        raise SystemExit(
            "Stopping before the 100-example train. The loss string must contain "
            '<|tool_call>call: and must not contain {"name". '
            "Refusing to train a plain-text adapter."
        )


def build_checked_tool_train_set(
    rows: list[dict], limit: int, count_tokens, render, n: int = TOOL_TRAIN_EXAMPLES
):
    """Render one example, require Gemma tool-call tokens, then take the rest."""
    probe, _probe_skipped = build_tool_train_set(rows, limit, count_tokens, render, n=1)
    require_tool_call_loss(probe[0]["text"])
    print(
        "Loss check passed on the first example: it contains <|tool_call>call: and not {\"name\".",
        flush=True,
    )
    if n <= 1:
        return probe, _probe_skipped
    return build_tool_train_set(rows, limit, count_tokens, render, n=n)


def clip_generation(text: str) -> str:
    """Keep through the first tool-call close. Drop a following tool result."""
    open_at = text.find(_CALL_OPEN)
    if open_at < 0:
        cut = text.find(_RESPONSE_OPEN)
        return text if cut < 0 else text[:cut]
    close_at = text.find(_CALL_CLOSE, open_at)
    if close_at < 0:
        cut = text.find(_RESPONSE_OPEN, open_at)
        return text[:cut] if cut >= 0 else text
    return text[: close_at + len(_CALL_CLOSE)]


def parse_gemma_arguments(body: str) -> dict:
    """Parse Gemma's call body: ``key:<|"|>text<|"|>`` or a bare number."""
    args: dict = {}
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i] in " \n\t,":
            i += 1
        if i >= n:
            break
        colon = body.find(":", i)
        if colon < 0:
            break
        key = body[i:colon].strip()
        i = colon + 1
        if body.startswith(_STRING, i):
            i += len(_STRING)
            end = body.find(_STRING, i)
            if end < 0:
                value = body[i:]
                i = n
            else:
                value = body[i:end]
                i = end + len(_STRING)
        else:
            end = i
            while end < n and body[end] != ",":
                end += 1
            raw = body[i:end].strip()
            i = end
            if raw == "true":
                value = True
            elif raw == "false":
                value = False
            elif raw in {"null", "none", ""}:
                value = None
            else:
                try:
                    value = int(raw)
                except ValueError:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw
        if key:
            args[key] = value
    return args


def parse_first_call(text: str) -> dict | None:
    """The first ``<|tool_call>call:name{...}<tool_call|>`` span, if it is complete."""
    start = text.find(_CALL_OPEN)
    if start < 0:
        return None
    i = start + len(_CALL_OPEN)
    name_end = i
    while name_end < len(text) and (text[name_end].isalnum() or text[name_end] == "_"):
        name_end += 1
    name = text[i:name_end]
    if not name:
        return None
    close_at = text.find(_CALL_CLOSE, name_end)
    if close_at < 0:
        return None
    head = text[name_end:close_at].strip()
    if head.startswith("("):
        quoted = re.search(r'"((?:\\.|[^"\\])*)"', head)
        arguments = {"pattern": quoted.group(1)} if quoted else {}
    elif head.startswith("{") and head.endswith("}"):
        inner = head[1:-1]
        arguments = _json_object(head) or parse_gemma_arguments(inner)
    else:
        arguments = parse_gemma_arguments(head)
    return {"name": name, "arguments": arguments}


def assistant_from_gemma_text(text: str) -> dict:
    """One harness message. A tool result after the call is not part of it."""
    thinking, visible = split_thinking(text or "")
    clipped = clip_generation(visible)
    parsed = parse_first_call(clipped)
    message: dict = {"role": "assistant", "content": "" if parsed else clipped.strip()}
    if thinking:
        message["reasoning"] = thinking
    if parsed is not None:
        message["tool_calls"] = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": parsed["name"],
                    "arguments": json.dumps(parsed["arguments"], ensure_ascii=False),
                },
            }
        ]
    return message


def next_turn(text: str, stopped: bool) -> tuple[dict, bool]:
    """Return one call. An unknown name is returned once, then the loop stops."""
    if stopped:
        return {"role": "assistant", "content": "Stopped."}, True
    message = assistant_from_gemma_text(text)
    calls = message.get("tool_calls") or []
    if calls and calls[0]["function"]["name"] not in TOOL_NAMES:
        return message, True
    return message, False


def prepare_for_template(messages: list[dict]) -> list[dict]:
    """Tool arguments must be mappings so the template emits Gemma's call syntax."""
    prepared = []
    for message in messages:
        item = dict(message)
        calls = []
        for call in item.get("tool_calls") or []:
            copied = dict(call)
            function = dict(copied.get("function") or {})
            raw = function.get("arguments")
            if isinstance(raw, str):
                try:
                    function["arguments"] = json.loads(raw)
                except json.JSONDecodeError:
                    function["arguments"] = {}
            elif not isinstance(raw, dict):
                function["arguments"] = {}
            copied["function"] = function
            calls.append(copied)
        if calls:
            item["tool_calls"] = calls
        prepared.append(item)
    return prepared


def completion_only(prompt: str, raw: str) -> str:
    if prompt and raw.startswith(prompt):
        return raw[len(prompt) :]
    return raw


def _json_object(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
