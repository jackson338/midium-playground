"""Clone a few hundred small, permissively licensed GitHub repos.

Search uses the public GitHub API. ``GITHUB_TOKEN`` is optional and lifts
the rate limit. Product trees (Midium, Courier, Unified Compute Engine)
are skipped.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import httpx

from playground.config import HOLDOUT_REPO_NAMES, REPOS_DIR, Settings

_LANGUAGES = ("Python", "TypeScript", "Go", "Rust", "Java", "Ruby")
_LICENSES = ("mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause")
# GitHub size is kilobytes. This band is a real codebase that still clones fast.
# Qualifiers are ANDed. GitHub search does not accept a parenthesized OR of
# license: qualifiers — that query returns total_count 0.
_SIZE = "size:400..12000"


def is_holdout(full_name: str) -> bool:
    owner, _, name = full_name.partition("/")
    folded = name.casefold().replace("_", "-")
    if folded in HOLDOUT_REPO_NAMES or name.casefold() in HOLDOUT_REPO_NAMES:
        return True
    if "midium" in folded or "unified-compute" in folded:
        return True
    if owner.casefold() in {"recursionai", "recursion-ai", "midium"}:
        return True
    return False


def clone_dir(full_name: str) -> Path:
    return REPOS_DIR / full_name.replace("/", "__")


def search_query(license_name: str, language: str) -> str:
    return (
        f"license:{license_name} fork:false archived:false {_SIZE} "
        f"language:{language} stars:15..30000"
    )


async def discover(cfg: Settings, wanted: int) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "midium-playground"}
    if cfg.github_token:
        headers["Authorization"] = f"Bearer {cfg.github_token}"
    found: dict[str, dict] = {}
    # Spread the set across languages instead of filling it from the first page of Python.
    per_language = max(1, (wanted + len(_LANGUAGES) - 1) // len(_LANGUAGES))
    async with httpx.AsyncClient(headers=headers, timeout=30) as client:
        for language in _LANGUAGES:
            if len(found) >= wanted:
                break
            before = len(found)
            for license_name in _LICENSES:
                if len(found) - before >= per_language or len(found) >= wanted:
                    break
                for page in (1, 2):
                    if len(found) - before >= per_language or len(found) >= wanted:
                        break
                    query = search_query(license_name, language)
                    items = await _search_page(client, query, page, has_token=bool(cfg.github_token))
                    _take_items(found, items, language, license_name, wanted, per_language, before)
    if len(found) < wanted:
        raise SystemExit(
            f"GitHub search returned {len(found)} repos, wanted {wanted}. "
            "The query ran, but not enough permissive repos came back."
        )
    return list(found.values())[:wanted]


async def _search_page(client: httpx.AsyncClient, query: str, page: int, *, has_token: bool) -> list[dict]:
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": 100, "page": page}
    response = await client.get("https://api.github.com/search/repositories", params=params)
    if response.status_code == 403:
        await asyncio.sleep(8)
        response = await client.get("https://api.github.com/search/repositories", params=params)
    if response.status_code == 403:
        hint = "" if has_token else " Set GITHUB_TOKEN in .env and retry."
        raise SystemExit(f"GitHub search was rate-limited.{hint}")
    if response.status_code >= 400:
        detail = response.text[:300].replace("\n", " ")
        raise SystemExit(f"GitHub search failed ({response.status_code}) for {query!r}: {detail}")
    payload = response.json()
    return payload.get("items") or []


def _take_items(
    found: dict[str, dict],
    items: list[dict],
    language: str,
    license_name: str,
    wanted: int,
    per_language: int,
    before: int,
) -> None:
    for item in items:
        if len(found) >= wanted or len(found) - before >= per_language:
            return
        full = item.get("full_name") or ""
        if not full or full in found or is_holdout(full):
            continue
        found[full] = {
            "full_name": full,
            "clone_url": item.get("clone_url"),
            "license": (item.get("license") or {}).get("spdx_id") or license_name,
            "stars": item.get("stargazers_count") or 0,
            "description": item.get("description") or "",
            "language": item.get("language") or language,
        }


async def clone_all(repos: list[dict], concurrency: int = 8) -> list[dict]:
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def one(repo: dict) -> dict | None:
        async with sem:
            return await asyncio.to_thread(_clone_one, repo)

    cloned = await asyncio.gather(*(one(repo) for repo in repos))
    ready = [repo for repo in cloned if repo is not None]
    if not ready:
        raise SystemExit("No repositories cloned.")
    return ready


def _clone_one(repo: dict) -> dict | None:
    dest = clone_dir(repo["full_name"])
    if not (dest / ".git").is_dir():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return None
        proc = subprocess.run(
            ["git", "clone", "--depth", "1", repo["clone_url"], str(dest)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
    commit = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0:
        return None
    repo = dict(repo)
    repo["path"] = str(dest)
    repo["commit"] = commit.stdout.strip()
    return repo


def catalog(root: Path, limit_files: int = 40) -> dict:
    """File list the commission writer sees. Long files are listed first."""
    rows: list[dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if any(part.startswith(".") or part in {"node_modules", "dist", "build", "target", "vendor"} for part in Path(rel).parts):
            continue
        if path.suffix.lower() not in {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".rb", ".c", ".h", ".cc", ".cpp", ".md"
        }:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size < 200 or size > 1_500_000:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        lines = text.count("\n") + 1
        if lines < 40:
            continue
        rows.append({"path": rel, "lines": lines, "bytes": size})
    rows.sort(key=lambda item: item["lines"], reverse=True)
    return {
        "longest": rows[:limit_files],
        "long_enough": sum(1 for item in rows if item["lines"] >= 1200),
    }
