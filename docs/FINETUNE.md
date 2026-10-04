# Fine-tune plan

Do not start the full train. The first three runs are memory probes only. The machine is the M3 Ultra Mac Studio, 512GB. The hard ceiling is 400GB of process physical footprint. The MacBook only generates data.

## What gets published

Push the playground code and the training traces. Do not push `data/repos/`, `.env`, tokens, or the SubSWE model traces. Training jsonl lives at `data/traces/` and is committed on purpose, so a `git clone` on the Studio is the whole setup. This file states the clone, the three probe commands, and that LoRA does not need Midium or OpenRouter credentials.

Base model is Gemma 4 E4B, text only. Freeze vision and audio. Train with Unsloth on MLX, bf16 LoRA, batch size 1. Do not write a custom trainer. QAT 4-bit is the ship target, not this probe. The probe measures LoRA memory. The 4-bit run comes after the map.

## Three probes, in order

Each probe is one step, batch size 1, one fixed example. Print unified memory before the step, at peak during the step, and after it. Write that to `data/probes/<name>.json` with context length, tokens in the example, peak memory, and whether it finished or was killed. Stop the run after the one step. A watcher aborts the process if its physical footprint crosses 400GB. Do not start the next probe if the previous one was killed or its peak crossed 400GB.

1. `probe-16k`. One trace trimmed or packed to under 16k tokens.
2. `probe-32k`. Same, under 32k.
3. `probe-96k`. One real trace packed to 96k by real `read_file` pages, not padding. Drop it if you cannot pack it under 96k. This is the number that decides the batch size.

Commands:

```
uv run python -m playground probe --context 16384
uv run python -m playground probe --context 32768
uv run python -m playground probe --context 98304
```

## After the map

The 32k probe is the train setting. One step at 30,449 tokens used 127GB of Metal memory and 155GB of process footprint, in 70.5 seconds (432 tokens per second). The 55,741-token step used 289GB of Metal memory and 335GB of footprint. A full 96k sequence does not fit under 400GB.

Train one epoch at a 32,768-token cap. About 62 of the 877 train episodes are over 32k; capping them leaves about 16.3 million tokens. Holdout rows are not trained.

- Batch size 1: about 160GB and 10 to 14 hours. This is the first run.
- Batch size 2: about 290GB and 12 to 20 hours. This is the largest batch under 400GB. It does not reliably finish sooner.
- Batch size 3: about 430GB. Refused.

```bash
git pull
uv run python -m playground train --context 32768 --batch-size 1
```

The adapter is written to `data/checkpoints/e4b-32k-lora`. A second epoch would roughly double the time. SubSWE eval of that adapter is a later step. The baseline is the Gemma 4 E4B SubSWE score. The trained model has to beat that score, then Laguna XS 2.1 (36/40 on run 1). Export a 4-bit checkpoint only after the bf16 LoRA beats the baseline. QAT is a second train, not a quantize of this run.

Do not change the reader prompt, do not regenerate the 5k, and do not launch the full train from the probe script.
