# Data layout

The Mac Studio LoRA job does not talk to Midium Cloud or OpenRouter. Copy the trace files onto the Studio next to a clone of this repo.

## On the MacBook

| Path | What it is |
|---|---|
| `data/repos/` | Shallow clones. Gitignored. Not needed on the Studio. |
| `data/commissions/commissions.jsonl` | Objectives Flash wrote. One JSON object per line: repo, commit, path, objective, line range. |
| `data/commissions/smoke.jsonl` | The 10 commissions used for every smoke teacher. |
| `data/traces/smoke_report.json` | Scores for Laguna S 2.1, Laguna XS 2.1, and Qwen3.8 27B. Nothing in this file picks a winner. |
| `data/traces/smoke/` | Smoke episodes, one JSONL per teacher. |
| `data/traces/<teacher>.jsonl` | Full set after you choose a teacher. |
| `data/traces/dry.jsonl` | Scripted dry run. No credentials. |

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
# copy the SSD's data/traces/ onto data/traces/ beside that clone
```

The train reads `data/traces/` only.
