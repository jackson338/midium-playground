# Data layout

The Mac Studio LoRA job does not talk to Midium Cloud or OpenRouter. A clone of this repo already contains the teacher traces.

## On the MacBook

| Path | What it is |
|---|---|
| `data/repos/` | Shallow clones. Gitignored. Not needed on the Studio. |
| `data/commissions/commissions.jsonl` | Objectives Flash wrote. One JSON object per line: repo, commit, path, objective, line range. |
| `data/commissions/smoke.jsonl` | The 2 commissions used for every smoke teacher. |
| `data/traces/smoke_report.json` | Scores for Laguna S 2.1, Laguna XS 2.1, and Qwen3.8 27B. Nothing in this file picks a winner. |
| `data/traces/smoke/` | Smoke episodes, one JSONL per teacher. |
| `data/traces/qwen3-8-flash-next.part1.jsonl` and `part2.jsonl` | Teacher set committed for the Studio, split so each file stays under GitHub's 100MB limit. |
| `data/traces/dry.jsonl` | Scripted dry run. No credentials. |
| `data/benchmarks/e4b-baseline.json` | Gemma 4 E4B reader baseline. Compare a later label with `benchmark compare`. |

## One episode

Each line is one reader commission:

- `id`, `repo`, `commit`, `teacher`, `commission`
- `tool_rounds`: name, arguments, and the raw tool result (including `read_file` page text)
- `thinking`: list of `{round, text}`. Never merged into the tool call or the report
- `report`: final assistant message
- `token_estimate`, `char_length`, `split` (`train` or `holdout`)
- `raw_turn`: the Midium message list

Episodes whose token estimate is over 96k are not written.

## Studio

```text
git clone <this repo>
uv sync --group studio
uv run python -m playground probe --context 16384
```

No SSD copy and no Midium or OpenRouter keys. Weights are not in the repo. The first probe downloads `unsloth/gemma-4-E4B-it` into the Hugging Face cache. The train reads `data/traces/qwen3-8-flash-next.part1.jsonl` and `part2.jsonl`.
