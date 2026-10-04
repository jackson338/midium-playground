"""HTTP clients for Midium Cloud and OpenRouter.

Midium Cloud is the OpenAI-compatible inference host. The base URL default
is ``INFERENCE_BASE_URL`` (``https://api.midium.dev/``) and the credential
is ``ScoutConfig.cloud_api_key``. The served model name is the cloud tag
stripped, the same way ``served_model_name`` works in UCE routing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from playground.config import Settings

# A 500 is retried on that same completion. Later attempts wait this long.
_SERVER_ERROR_DELAYS = (2.0, 8.0, 20.0)


class CloudError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


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
        response = await self._post_with_server_retries(body)
        if response.status_code >= 400:
            raise CloudError(
                f"Midium Cloud {response.status_code}: {response.text[:500]}",
                status=response.status_code,
            )
        return _assistant_message(response.json(), "Midium Cloud")

    async def _post_with_server_retries(self, body: dict[str, Any]) -> httpx.Response:
        return await post_retrying_500(
            self._client,
            self._chat_path(),
            body,
            label=f"Midium Cloud on {self.model}",
        )


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

    async def complete(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """Tool-calling completion used when this model is the reader teacher."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        response = await post_retrying_500(
            self._client,
            "chat/completions",
            body,
            label="OpenRouter",
        )
        if response.status_code >= 400:
            raise CloudError(
                f"OpenRouter {response.status_code}: {response.text[:500]}",
                status=response.status_code,
            )
        return _assistant_message(response.json(), "OpenRouter")

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


async def post_retrying_500(
    client: httpx.AsyncClient,
    path: str,
    body: dict[str, Any],
    *,
    label: str,
) -> httpx.Response:
    """Retry an HTTP 500 on the same completion. Other statuses return immediately."""
    delays = _SERVER_ERROR_DELAYS
    response: httpx.Response | None = None
    for attempt, delay in enumerate((0.0, *delays), start=1):
        if delay:
            print(f"  {label} 500, retry {attempt - 1} in {delay:.0f}s", flush=True)
            await asyncio.sleep(delay)
        response = await client.post(path, json=body)
        if response.status_code != 500 or attempt > len(delays):
            return response
    assert response is not None
    return response


def _assistant_message(payload: dict, label: str) -> dict:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CloudError(f"{label} returned no assistant message: {payload!r}") from exc
    if not isinstance(message, dict):
        raise CloudError(f"{label} assistant message was not an object")
    message["_raw"] = payload
    message["_usage"] = payload.get("usage") or {}
    return message


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
