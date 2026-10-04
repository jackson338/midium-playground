# SubSWE

SubSWE scores a commissioned code reader. The model gets a public repo at a pinned commit, one question, and the read-only reader tools (`match_path`, `get_cwd`, `change_directory`, `list_files`, `glob`, `grep`, `read_file`, `os_bash`, `web_search`, `fetch_url`). The final assistant text is the report. There is no submit tool and no write tool. A leaked tool-call token is not a report.

This is not SWE-bench. There are no fail-to-pass tests, no patch, and no issue to close.

## Published number

The published number is a deterministic pass rate. A task passes only when every check that applies to its kind is true. A model judge may add a good or bad column. That column does not move the headline number. A failed judge call is reported separately and stays in the denominator.

A reader cloud error is invalid. It is counted in `errors` and left out of the rate.

`subswe --report` rescores the trace files. It does not score only the rows written by the latest process. It refuses a published rate if Gemma 4 E4B, Qwen3.8 Flash Next, or Laguna XS 2.1 is missing. Two runs are printed separately. They are not averaged.

## Tasks

`benchmarks/subswe.json` holds 40 tasks across at least eight permissive public repos (MIT, Apache-2.0, or BSD). Commits are pinned. Unified Compute Engine, Courier, Midium, and the owners `recursionai`, `recursion-ai`, `midium`, and `jacksonoaks` are excluded. Those trees are also held out of teacher traces.

Kinds:

- `locate`: name the file that defines a symbol.
- `read`: open that file and quote a gold substring.
- `paraphrase`: find the file from a description that does not hand over the path.
- `negative`: the symbol is absent. A pass needs a `grep` or `read_file` that uses that symbol, plus an absence claim. A `glob` of `**/*.py` is not that search.
- `multi-file`: name and read both gold paths.
- `long-page`: the gold quote sits past line 500, so one short page fails.
- `distractor`: a similarly named file is the wrong answer and must not be the named path.

The eight Unified Compute Engine tasks live in `benchmarks/uce-smoke.json`. They are private smoke and are not part of the published rate.

## Checks

- `valid_tools`: every call name is in the reader set, required arguments are present, and `read_file` line numbers are integers. A numeric string is a fail even if the tool would coerce it.
- `stopped_clean`: at most 16 tool calls, the run ended with a report, and the report is not a tool-call token (`<|tool_call>` or a bare `call:`). Calls are counted, not the shared round index. The reader loop for SubSWE allows 16 iterations. Teacher traces stay at 8.
- `named_paths`: the report names each gold path by the full relative path or its last two components.
- `read_page`: when `must_read` is set, a `read_file` result covers the gold quote and the report contains that exact substring.
- `symbol_seen`: the gold symbol is in the report, unless the task expects it to be absent.
- `negative_search`: for an absent symbol, some `grep` or `read_file` used that symbol and the report claims it is absent.
- `no_false_absence`: if the quote is in the file, the report must not say it is missing.

`grep` accepts a file or a directory. A file searches that file.

## Commands

```bash
uv run python -m playground subswe --model "Gemma 4 E4B" --run 1
uv run python -m playground subswe --model "Qwen3.8 Flash Next" --run 1
uv run python -m playground subswe --model "Laguna XS 2.1" --run 1
uv run python -m playground subswe --report
```

Repeat with `--run 2`. Traces go to `data/benchmarks/subswe/<model>.run<N>.jsonl`. The score file is `data/benchmarks/subswe-report.json`. Local Gemma 4 E4B runs one task at a time. Laguna XS 2.1 stays at concurrency 2. Qwen3.8 Flash Next on OpenRouter runs at concurrency 8.

## Fine-tuned model on the Studio

The comparison is F16 Gemma 4 E4B with no adapter, then a 100-example tool-call LoRA at `data/checkpoints/e4b-32k-tools`. The old `e4b-32k-lora` adapter is not this run. The eight task repos are not in the clone. Grading stays off, so no Midium key is required.

```bash
uv run python -m playground compare-lora
```

That fetches the pinned repos when they are missing, writes `gemma-4-e4b-f16.run1.jsonl`, trains, deletes `gemma-4-e4b-lora.run1.jsonl` so the old 1/40 file cannot be resumed, then writes a new LoRA trace. It prints both pass rates, per-kind rates, call counts, and the checks that failed.

`--score` still prints one file's pass rate. It does not require the other models' traces.
