import asyncio

import httpx

from playground.cloud import MidiumCloud
from playground.config import Settings


def _settings() -> Settings:
    return Settings(
        openrouter_api_key="",
        openrouter_model="qwen/qwen3.8-flash",
        openrouter_base_url="https://openrouter.ai/api/v1",
        midium_api_key="test",
        midium_base_url="https://example.test/",
        local_api_key="",
        local_base_url="",
        brave_api_key="",
        github_token="",
    )


def test_midium_retries_http_500_then_returns_the_page(monkeypatch):
    monkeypatch.setattr("playground.cloud._SERVER_ERROR_DELAYS", (0.0, 0.0, 0.0))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(500, json={"detail": "internal"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    async def run() -> dict:
        cloud = MidiumCloud(_settings(), "Laguna XS 2.1")
        await cloud.aclose()
        cloud._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://example.test/",
        )
        try:
            return await cloud.complete([{"role": "user", "content": "hi"}], None)
        finally:
            await cloud.aclose()

    message = asyncio.run(run())
    assert calls["n"] == 3
    assert message["content"] == "ok"


def test_midium_does_not_retry_a_client_error(monkeypatch):
    monkeypatch.setattr("playground.cloud._SERVER_ERROR_DELAYS", (0.0, 0.0, 0.0))
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    async def run() -> None:
        cloud = MidiumCloud(_settings(), "Laguna XS 2.1")
        await cloud.aclose()
        cloud._client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://example.test/",
        )
        try:
            await cloud.complete([{"role": "user", "content": "hi"}], None)
        finally:
            await cloud.aclose()

    try:
        asyncio.run(run())
    except Exception as exc:
        assert getattr(exc, "status", None) == 400
    else:
        raise AssertionError("expected a 400")
    assert calls["n"] == 1
