"""One-epoch Gemma 4 E4B LoRA at a 32k token cap.

This is not the probe. Batch size 1 is the measured setting, about 160GB.
Batch size 2 is the largest that stays under the 400GB footprint ceiling.
"""

from __future__ import annotations

import os
import signal
import threading
from pathlib import Path

from playground.config import ROOT
from playground.probe import (
    MEMORY_CEILING_BYTES,
    MODEL_NAME,
    CountTokens,
    load_teacher_rows,
    memory_pressure,
    pack_example,
    phys_footprint,
)

TRAIN_CONTEXT = 32768
MAX_BATCH_SIZE = 2
CHECKPOINT_DIR = ROOT / "data" / "checkpoints" / "e4b-32k-lora"


def validate_train_args(context: int, batch_size: int) -> None:
    if context != TRAIN_CONTEXT:
        raise SystemExit(
            f"This train is capped at {TRAIN_CONTEXT} tokens. Refusing context {context}."
        )
    if batch_size not in (1, 2):
        raise SystemExit(
            f"Batch size {batch_size} is not allowed. Use 1, or 2 to fill the 400GB budget."
        )


def build_train_set(rows: list[dict], limit: int, count_tokens: CountTokens) -> tuple[list[dict], int]:
    """Train rows only, each packed to ``limit`` tokens. Holdout rows are skipped."""
    packed: list[dict] = []
    skipped = 0
    for row in rows:
        if row.get("split") != "train":
            skipped += 1
            continue
        item = pack_example(row, limit, count_tokens)
        if item is None or item["tokens"] > limit:
            skipped += 1
            continue
        packed.append(item)
    if not packed:
        raise SystemExit("No train episodes fit under the 32k cap.")
    return packed, skipped


def run_train_cli(context: int, batch_size: int) -> None:
    validate_train_args(context, batch_size)
    from playground.probe import _load_tokenizer, _require_unsloth

    rows = load_teacher_rows()
    count_tokens = _load_tokenizer()
    examples, skipped = build_train_set(rows, context, count_tokens)
    tokens = sum(item["tokens"] for item in examples)
    print(
        f"Train set: {len(examples)} episodes, {tokens} tokens, skipped {skipped}, "
        f"batch_size={batch_size}",
        flush=True,
    )
    _require_unsloth()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    peak = {"bytes": phys_footprint()}
    outcome = {"status": "finished"}

    def watch() -> None:
        while not stop.wait(0.5):
            used = phys_footprint()
            peak["bytes"] = max(peak["bytes"], used)
            if used >= MEMORY_CEILING_BYTES:
                outcome["status"] = "killed"
                _write_status("killed", examples=len(examples), tokens=tokens, peak_bytes=used)
                os.kill(os.getpid(), signal.SIGTERM)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        _unsloth_train([item["text"] for item in examples], context, batch_size)
    finally:
        stop.set()
        after = phys_footprint()
        peak["bytes"] = max(peak["bytes"], after)
        if outcome["status"] != "killed":
            _write_status("finished", examples=len(examples), tokens=tokens, peak_bytes=peak["bytes"])
        print(
            f"Saved {CHECKPOINT_DIR} status={outcome['status']} "
            f"peak_bytes={peak['bytes']} pressure={memory_pressure()}",
            flush=True,
        )


def _unsloth_train(texts: list[str], max_seq_length: int, batch_size: int) -> None:
    from unsloth import FastModel
    from transformers import TrainingArguments

    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        full_finetuning=False,
    )
    lora = dict(
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )
    try:
        model = FastModel.get_peft_model(model, finetune_audio_layers=False, **lora)
    except TypeError:
        model = FastModel.get_peft_model(model, **lora)
    from trl import SFTTrainer

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=batch_size,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    dataset = [{"text": text} for text in texts]
    try:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=args,
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=args,
        )
    trainer.train()
    trainer.save_model(str(CHECKPOINT_DIR))


def _write_status(status: str, *, examples: int, tokens: int, peak_bytes: int) -> None:
    import json

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "context": TRAIN_CONTEXT,
        "examples": examples,
        "tokens": tokens,
        "peak_bytes": peak_bytes,
        "ceiling_bytes": MEMORY_CEILING_BYTES,
    }
    (CHECKPOINT_DIR / "train-status.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
