# Midium playground

This repo generates reader training traces on the MacBook. The Gemma 4 E4B LoRA is out of scope for this change. It runs later on the Mac Studio and only reads `data/traces/`. It does not need Midium or OpenRouter credentials.

The reader loop is the Unified Compute Engine research junior (public name reader, internal researcher): the same tools, the same 6-round wrap-up, and a hard stop at 8 rounds. The report is the final assistant message. `read_file` is the one change. In UCE the research junior summarizes files longer than about 400 lines. Here `read_file` takes `start_line` and `end_line` and returns that page of raw lines.

## Setup

```bash
uv sync
```

`.env` is created with empty values. Fill it from `.env.example`:

- `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (`qwen/qwen3.8-flash`, the hosted Qwen3.8 Flash / Flash-Next)
- `MIDIUM_CLOUD_API_KEY` and `MIDIUM_CLOUD_BASE_URL` (`https://api.midium.dev/`, from UCE `INFERENCE_BASE_URL`; the key is `ScoutConfig.cloud_api_key`)
- `BRAVE_SEARCH_API_KEY` for the reader's `web_search`
- `GITHUB_TOKEN` optional, for cloning a few hundred repos

## Commands

Dry run, no keys. Writes two valid episodes to `data/traces/dry.jsonl`.

```bash
uv run python -m playground dry-run
```

Smoke. Same 10 commissions on Laguna S 2.1, Laguna XS 2.1, and Qwen3.8 27B. Writes `data/traces/smoke_report.json` and does not pick a winner.

```bash
uv run python -m playground smoke
```

Commissions. Clones about 200 permissive repos and asks Flash for 25 objectives in each (about 5,000). Repo calls and model calls run concurrently.

```bash
uv run python -m playground commissions --repos 200 --per-repo 25 --concurrency 16
```

Full set, after you choose a teacher from the smoke report. `--teacher` is required. Anything over 96k tokens is dropped.

```bash
uv run python -m playground generate --teacher "Laguna S 2.1" --n 5000
```

The other served names are `Laguna XS 2.1` and `Qwen3.8 27B`.

## Studio handoff

See `docs/DATA_LAYOUT.md`. Code is the git clone. Traces are what you copy on an SSD to `data/traces/`.
