"""web_search and fetch_url, matching the UCE reader tools.

web_search reads BRAVE_SEARCH_API_KEY, the same name as
``src/web_search/brave_client.py``. fetch_url follows
``src/scout/tools/fetch.py``: browser-like headers, GitHub blob rewrite,
HTML stripped to text, bot-challenge pages returned as errors.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
_BROWSER_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT_SECONDS = 10.0
_MAX_RESPONSE_BYTES = 5 * 1024 * 1024
_BLOCK_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "cf-challenge",
    "enable javascript and cookies to continue",
    "attention required! | cloudflare",
    "access denied",
    "you have been blocked",
    "please verify you are a human",
    "checking your browser before accessing",
    "captcha-delivery.com",
    "px-captcha",
)
_GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<ref>[^/]+)/(?P<path>.+)$"
)
_SKIP_TAGS = {"script", "style", "noscript", "template"}
_BLOCK_TAGS = {
    "p", "br", "div", "section", "article", "header", "footer", "nav",
    "li", "ul", "ol", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "pre", "blockquote", "hr",
}


def web_search(query: str, count: int = 5) -> dict:
    if not isinstance(query, str) or not query.strip():
        return {"error": "query is required and must be a non-empty string."}
    try:
        count_n = int(count)
    except (TypeError, ValueError):
        count_n = 5
    count_n = max(1, min(count_n, 10))
    key = os.environ.get("BRAVE_SEARCH_API_KEY", "").strip()
    if not key:
        return {"error": "Web search is not configured on this server (BRAVE_SEARCH_API_KEY missing)."}
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": key,
            },
            params={"q": query.strip(), "count": count_n},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Search failed: {exc!r}"}
    results = []
    for item in data.get("web", {}).get("results", [])[:count_n]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "description": item.get("description", ""),
            }
        )
    return {
        "query": query.strip(),
        "results": results,
        "formatted": _format_search_results(query.strip(), results),
    }


def _format_search_results(query: str, results: list[dict], max_desc_chars: int = 300) -> str:
    if not results:
        return f'No web results found for "{query}".'
    lines = [f'Web search results for "{query}":\n']
    for i, item in enumerate(results, 1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        desc = item.get("description", "")
        if len(desc) > max_desc_chars:
            desc = desc[:max_desc_chars].rsplit(" ", 1)[0] + "..."
        lines.append(f"{i}. **{title}** ({url})\n   {desc}\n")
    return "\n".join(lines)


def fetch_url(url: str, max_chars: int = 20000) -> dict:
    if not isinstance(url, str) or not url.strip():
        return {"url": url, "error": "url is required and must be a non-empty string."}
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return {"url": url, "error": f"Unsupported URL scheme '{parsed.scheme}'. Use http or https."}
    try:
        max_chars_n = int(max_chars)
    except (TypeError, ValueError):
        max_chars_n = 20000
    max_chars_n = max(500, min(max_chars_n, 200_000))
    target = _rewrite_github_blob(url)
    try:
        resp = httpx.get(target, timeout=_TIMEOUT_SECONDS, follow_redirects=True, headers=_BROWSER_HEADERS)
    except httpx.HTTPError as exc:
        return {"url": url, "final_url": target, "error": f"Request failed: {exc!r}"}
    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower() or "application/octet-stream"
    body_bytes = resp.content[:_MAX_RESPONSE_BYTES]
    if not _is_textlike(content_type):
        return {
            "url": url,
            "final_url": str(resp.url),
            "status": resp.status_code,
            "content_type": content_type,
            "error": f"Unsupported content type '{content_type}'. fetch_url returns text only.",
        }
    encoding = resp.encoding or "utf-8"
    try:
        body = body_bytes.decode(encoding, errors="replace")
    except LookupError:
        body = body_bytes.decode("utf-8", errors="replace")
    if _looks_blocked(resp.status_code, body):
        return {
            "url": url,
            "final_url": str(resp.url),
            "status": resp.status_code,
            "content_type": content_type,
            "blocked": True,
            "error": (
                f"This site blocked the request (HTTP {resp.status_code}) — it "
                "serves a bot-protection / CAPTCHA challenge instead of the page. "
                "Fetching won't get past it. Use the web_search snippet for this "
                "result, try a different source, or tell the user the page is gated."
            ),
        }
    title = None
    if content_type in {"text/html", "application/xhtml+xml"}:
        extractor = _TextExtractor()
        try:
            extractor.feed(body)
        except Exception:  # noqa: BLE001
            pass
        text = extractor.text() or body
        title = extractor.title
    else:
        text = body
    char_count = len(text)
    truncated = char_count > max_chars_n
    if truncated:
        text = text[:max_chars_n]
    return {
        "url": url,
        "final_url": str(resp.url),
        "status": resp.status_code,
        "content_type": content_type,
        "title": title,
        "content": text,
        "truncated": truncated,
        "char_count": char_count,
    }


def _looks_blocked(status: int, body: str) -> bool:
    if status in (401, 403, 429, 503):
        return True
    head = body[:4000].lower()
    return any(marker in head for marker in _BLOCK_MARKERS)


def _rewrite_github_blob(url: str) -> str:
    match = _GITHUB_BLOB_RE.match(url)
    if not match:
        return url
    return (
        "https://raw.githubusercontent.com/"
        f"{match.group('owner')}/{match.group('repo')}/{match.group('ref')}/{match.group('path')}"
    )


def _is_textlike(content_type: str) -> bool:
    ct = content_type.lower()
    return ct.startswith("text/") or any(token in ct for token in ("json", "xml", "yaml", "javascript"))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):  # type: ignore[override]
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str):  # type: ignore[override]
        if self._skip_depth > 0:
            return
        if self._in_title:
            self.title = data.strip() if self.title is None else (self.title + data).strip()
            return
        self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n[ \t]+", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()
