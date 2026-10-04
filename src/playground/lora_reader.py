"""Turn the fine-tuned Gemma 4 E4B LoRA into the reader loop's completer.

The harness is still run_reader: the research skill and the read-only tools.
The model was trained to write an assistant turn as {"name","arguments"}.
That object becomes a tool call. Any other text is the report.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from playground.harness.loop import split_thinking
from playground.probe import MODEL_NAME, enforce_metal_ceiling

ADAPTER_DIR = Path("data/checkpoints/e4b-32k-lora")


def transcript_from_loop_messages(messages: list[dict]) -> str:
    """Rebuild the packed training transcript, then open the next assistant turn."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role") or "assistant"
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                fn = call.get("function") or {}
                raw = fn.get("arguments") or "{}"
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw if isinstance(raw, dict) else {}
                payload = {"name": fn.get("name") or "", "arguments": args}
                lines.append("assistant\n" + json.dumps(payload, ensure_ascii=False))
            continue
        lines.append(f"{role}\n{message.get('content') or ''}")
    lines.append("assistant")
    return "\n".join(lines) + "\n"


def assistant_message_from_text(text: str) -> dict:
    """Map a generation to the OpenAI message run_reader already parses."""
    thinking, visible = split_thinking(text or "")
    payload = _tool_payload(visible)
    message: dict = {"role": "assistant", "content": "" if payload else visible}
    if thinking:
        message["reasoning"] = thinking
    if payload is not None:
        message["tool_calls"] = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": json.dumps(payload["arguments"], ensure_ascii=False),
                },
            }
        ]
    return message


def _tool_payload(visible: str) -> dict | None:
    text = visible.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    if not text.startswith("{"):
        return None
    try:
        parsed, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    if text[end:].strip() or not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    arguments = parsed.get("arguments")
    if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
        return None
    return {"name": name.strip(), "arguments": arguments}


class LoraReader:
    def __init__(self, adapter: Path):
        if not adapter.is_dir():
            raise SystemExit(f"LoRA adapter not found: {adapter}")
        self.adapter = adapter
        self._model = None
        self._processor = None

    async def aclose(self) -> None:
        self._model = None
        self._processor = None

    async def complete(self, messages: list[dict], tools: list[dict] | None) -> dict:
        del tools
        return await asyncio.to_thread(self._complete_sync, messages)

    def _complete_sync(self, messages: list[dict]) -> dict:
        if self._model is None:
            self._model, self._processor = load_lora(self.adapter)
        prompt = transcript_from_loop_messages(messages)
        raw = generate_continuation(self._model, self._processor, prompt)
        return assistant_message_from_text(raw)


def load_lora(adapter: Path):
    from playground.probe import _require_unsloth

    _require_unsloth()
    from mlx_vlm import load

    model, processor = load(MODEL_NAME, adapter_path=str(adapter))
    enforce_metal_ceiling()
    return model, processor


def generate_continuation(model, processor, prompt: str) -> str:
    from mlx_vlm import generate

    enforce_metal_ceiling()
    out = generate(model, processor, prompt=prompt, max_tokens=512, temperature=0.0, verbose=False)
    if isinstance(out, str):
        return out
    text = getattr(out, "text", None)
    return text if isinstance(text, str) else str(out)
