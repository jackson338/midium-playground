# Reader data generation

Scope of this repo right now: generate training traces for a Gemma 4 E4B reader LoRA.

Where it runs:

- This pipeline (clone repos, write commissions, run the reader, write JSONL) runs on the MacBook.
- The LoRA train runs later on the Mac Studio. It only reads the traces. It is not part of this change.
- Do not spend GCP credits. Do not invent Midium API URLs. Read them out of the UCE client.

## Outcome

A uv project that, on a clean MacBook:

1. Loads `.env` (empty file committed as a template; real secrets stay local).
2. Pulls small permissively licensed open-source repos.
3. Asks Qwen3.8 Flash Next on OpenRouter for reader commissions.
4. Runs those commissions through a copied UCE reader loop on Midium Cloud.
5. Writes portable JSONL under `data/traces/`.

Smoke: the same 10 commissions on Laguna S 2.1, Laguna XS 2.1, and Qwen3.8 27B, plus a comparison report. Do not auto-pick a winner.

After a teacher is chosen: about 5,000 episodes from that teacher. A substantial fraction of transcripts must be long enough to train at a 96k context (real file pages, not padding). Drop anything over 96k.

## Harness

Copy the reader from `unified_compute_engine`, then verify against the code. Do not trust this note if the code disagrees.

- Public name `reader`, internal `researcher`.
- Tools: `match_path`, `get_cwd`, `change_directory`, `list_files`, `glob`, `grep`, `read_file(path, offset, limit)`, read-only `os_bash`, `web_search`, `fetch_url`.
- The report is the final assistant message. There is no submit tool.
- Nudge to stop around 6 tool rounds. Hard cap 8.
- `read_file` is a pager and returns raw lines. Do not copy the over-400-line digest path.

Store the raw Midium turn and a normalized trace. Gemma-native tool calls are the later LoRA format. Gemma 4 E4B is a thinking model: keep thinking in its own field, never flattened into the tool call or the report.

## Layout

- `src/playground/` runner
- `data/repos/` clones, gitignored
- `data/traces/` JSONL, gitignored
- `docs/DATA_LAYOUT.md` SSD handoff
- `.env.example` with `OPENROUTER_API_KEY` and the Midium Cloud base URL and credential names found in UCE

Cursor creates `.env` with empty values. No real keys in git.

## Commissions

Flash Next writes objectives only. It does not run the reader. Each objective is one slice of one repo (find a symbol, trace a call, explain a single region). No whole-repo audits. Record the repo commit.

## Smoke

Same 10 commissions on Laguna S 2.1, Laguna XS 2.1, and Qwen3.8 27B via Midium. Score valid tool names and args, paths that exist at the cited lines, stop by round 8, and a report consistent with the tool results. Write `data/traces/smoke_report.json`.

## Full set

`--teacher` is required. Target 5,000 commissions. Pack real `read_file` pages so many transcripts land between 32k and 96k tokens. Hold out any repo that is one of our product trees. One JSONL row per episode: id, repo, commit, teacher, commission, tool rounds (name, args, raw page), thinking, final report, token estimate, char length, split.

## MacBook commands

- `uv sync`
- fill `.env`
- `uv run python -m playground smoke`
- `uv run python -m playground generate --teacher <name> --n 5000`

## Studio handoff

Code is what you git clone. Traces are what you copy on an SSD to `data/traces/` beside that clone. The later LoRA reads that directory and does not need Midium or OpenRouter credentials.

## Done

Smoke report exists. A few-episode dry run writes valid JSONL. `.env.example` is complete. README says the LoRA is out of scope for this change and will run on the Studio.
