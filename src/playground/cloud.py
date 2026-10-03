"""HTTP clients for Midium Cloud and OpenRouter.

Midium Cloud is the OpenAI-compatible inference host. The base URL default
is ``INFERENCE_BASE_URL`` (``https://api.midium.dev/``) and the credential
is ``ScoutConfig.cloud_api_key``. The served model name is the cloud tag
stripped, the same way ``served_model_name`` works in UCE routing.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from playground.config import Settings


class CloudError(RuntimeError):
    pass


def served_model_name(model: str) -> str:
    raw = (model or "").strip()
    if raw.startswith("cloud:"):
        return raw[len("cloud:") :]
    return raw


class MidiumCloud:
    def __init__(
        self,
        cfg: Settings,
        model: str,
        timeout: float = 180.0,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = served_model_name(model)
        key = cfg.midium_api_key if api_key is None else api_key
        base = cfg.midium_base_url if base_url is None else base_url
        self.base = base.rstrip("/") + "/"
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base,
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )

    def _chat_path(self) -> str:
        # OpenAI clients are given either the host or a base that already ends in /v1.
        if self.base.rstrip("/").endswith("/v1"):
            return "chat/completions"
        return "v1/chat/completions"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, messages: list[dict], tools: list[dict] | None) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "stream": False,
            # Same extra_body keys scout_relay_extra_body sends when the
            # thinking toggle is unset: stream reasoning if the model emits
            # it, and keep web search on the client (this harness).
            "web_search_client_side": True,
            "stream_reasoning": True,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        response = await self._client.post(self._chat_path(), json=body)
        if response.status_code >= 400:
            raise CloudError(f"Midium Cloud {response.status_code}: {response.text[:500]}")
        payload = response.json()
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CloudError(f"Midium Cloud returned no assistant message: {payload!r}") from exc
        if not isinstance(message, dict):
            raise CloudError("Midium Cloud assistant message was not an object")
        message["_raw"] = payload
        usage = payload.get("usage") or {}
        message["_usage"] = usage
        return message


class OpenRouter:
    def __init__(self, cfg: Settings, timeout: float = 120.0):
        self.model = cfg.openrouter_model
        self._client = httpx.AsyncClient(
            base_url=cfg.openrouter_base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {cfg.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete_text(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
        }
        response = await self._client.post("chat/completions", json=body)
        if response.status_code >= 400:
            raise CloudError(f"OpenRouter {response.status_code}: {response.text[:500]}")
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise CloudError(f"OpenRouter returned no content: {payload!r}") from exc
        return content or ""


def parse_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise CloudError("model did not return a JSON object")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise CloudError(f"commission JSON failed to parse: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CloudError("commission JSON was not an object")
    return parsed
