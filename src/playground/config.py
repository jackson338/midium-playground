"""Environment loaded from the repo-root .env. No secrets have defaults except public URLs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# INFERENCE_BASE_URL in unified_compute_engine/src/product.py
MIDIUM_INFERENCE_BASE_URL = "https://api.midium.dev/"

ROOT = Path(__file__).resolve().parents[2]
REPOS_DIR = ROOT / "data" / "repos"
TRACES_DIR = ROOT / "data" / "traces"
COMMISSIONS_PATH = ROOT / "data" / "commissions" / "commissions.jsonl"
SMOKE_COMMISSIONS_PATH = ROOT / "data" / "commissions" / "smoke.jsonl"
SMOKE_REPORT_PATH = TRACES_DIR / "smoke_report.json"

# Served names on Midium Cloud (cloud: tag stripped). Smoke runs all three.
TEACHERS: tuple[str, ...] = (
    "Laguna S 2.1",
    "Laguna XS 2.1",
    "Qwen3.8 27B",
)

# Hard context the later LoRA will train at. Episodes above this are dropped.
MAX_CONTEXT_TOKENS = 96_000
# Target band for a substantial fraction of the full set.
LONG_TRACE_MIN_TOKENS = 32_000

RESEARCH_WRAPUP_ITER = 6
RESEARCH_MAX_ITERS = 8

# Our product trees. A GitHub repo whose name matches is never cloned.
HOLDOUT_REPO_NAMES = frozenset(
    {
        "unified_compute_engine",
        "unified-compute-engine",
        "midium",
        "midium-playground",
        "courier",
        "courier-os",
        "courier-dashboard",
        "courier-dashboard-sh",
    }
)


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_env() -> None:
    _load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    openrouter_model: str
    openrouter_base_url: str
    midium_api_key: str
    midium_base_url: str
    brave_api_key: str
    github_token: str

    def require_openrouter(self) -> None:
        if not self.openrouter_api_key:
            raise SystemExit("Set OPENROUTER_API_KEY in .env before generating commissions.")

    def require_midium(self) -> None:
        if not self.midium_api_key:
            raise SystemExit("Set MIDIUM_CLOUD_API_KEY in .env before running a teacher.")


def settings() -> Settings:
    load_env()
    base = os.environ.get("MIDIUM_CLOUD_BASE_URL", "").strip() or MIDIUM_INFERENCE_BASE_URL
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "").strip() or "qwen/qwen3.8-flash",
        openrouter_base_url=os.environ.get("OPENROUTER_BASE_URL", "").strip()
        or "https://openrouter.ai/api/v1",
        midium_api_key=os.environ.get("MIDIUM_CLOUD_API_KEY", "").strip(),
        midium_base_url=base.rstrip("/") + "/",
        brave_api_key=os.environ.get("BRAVE_SEARCH_API_KEY", "").strip(),
        github_token=os.environ.get("GITHUB_TOKEN", "").strip(),
    )
