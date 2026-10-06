"""Distill a 4-bit Gemma 4 VLM from a fused bf16 teacher.

mlx-lm's DWQ command only loads text models. This runs the same KL loss
on the language, vision, and audio towers of the fused VLM, and keeps the
4-bit scales from the lowest validation point. Vision and audio are
quantized. The calibration text does not teach them.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import signal
import threading
from pathlib import Path

DWQ_BITS = 4
DWQ_GROUP_SIZE = 32
DWQ_NUM_SAMPLES = 1024
DWQ_VALID_SAMPLES = 32
DWQ_MAX_SEQ_LENGTH = 512
DWQ_TEMPERATURE = 2.0
DWQ_LEARNING_RATE = 1e-6
DWQ_BATCH_SIZE = 1
DWQ_SEED = 123
DWQ_DATASET = "allenai/tulu-3-sft-mixture"


def lower_validation(best: tuple[int, float] | None, step: int, loss: float) -> tuple[int, float]:
    """Keep the lowest validation loss. Step 0 is the untouched 4-bit scales."""
    if best is None or loss < best[1]:
        return step, loss
    return best


def accept_distillation(initial_loss: float, final_loss: float) -> None:
    """A worse validation loss must not be published as the reader."""
    if final_loss > initial_loss:
        raise SystemExit(
            f"DWQ validation loss got worse ({initial_loss:.4f} -> {final_loss:.4f}). "
            "Not scoring the 4-bit model."
        )


def validation_losses(lines: list[str]) -> tuple[float, float]:
    """Read the Validation lines mlx-lm prints during DWQ."""
    found: list[float] = []
    for line in lines:
        if "Validation:" not in line or "loss=" not in line:
            continue
        found.append(float(line.split("loss=", 1)[1].split(",", 1)[0]))
    if len(found) < 2:
        raise SystemExit("DWQ did not report a starting and final validation loss.")
    return found[0], found[-1]


def targets_ready(path: Path) -> bool:
    """A finished teacher pass wrote both splits for this batch and sequence length."""
    marker = path / "settings.json"
    if not marker.is_file():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if (
        saved.get("batch_size") != DWQ_BATCH_SIZE
        or saved.get("max_seq_length") != DWQ_MAX_SEQ_LENGTH
        or saved.get("seed") != DWQ_SEED
    ):
        return False
    return any((path / "train").glob("*.safetensors")) and any((path / "valid").glob("*.safetensors"))


def run_vlm_dwq(teacher_dir: Path, dest: Path) -> None:
    """Quantize ``teacher_dir`` and save a 4-bit VLM at ``dest``.

    The saved scales are the lowest validation point. That is the untouched
    4-bit start when training does not improve it.
    """
    import mlx.nn as nn

    from playground.probe import enforce_metal_ceiling

    target_dir = dest.parent / f"{dest.name}-targets"
    stop = threading.Event()
    threading.Thread(target=_watch_ceiling, args=(stop,), daemon=True).start()
    try:
        tokenizer = _teacher_pass(teacher_dir, target_dir)
        _release_mlx()
        student_vlm = _student_pass(nn, teacher_dir)
        student = _TextLogits(student_vlm)
        train_data, valid_data = _load_calibration(tokenizer)
        enforce_metal_ceiling()
        try:
            _checkpoint_layers(student.layers)
            initial, best, best_step = distill_scales(
                student,
                disk_target(target_dir),
                train_data,
                valid_data,
            )
        finally:
            enforce_metal_ceiling()
        accept_distillation(initial, best)
        _save_quantized(student_vlm, teacher_dir, dest, initial, best, best_step)
        print(
            f"DWQ 4-bit saved to {dest} (loss {initial:.4f} -> {best:.4f} at step {best_step})",
            flush=True,
        )
    finally:
        stop.set()


def _teacher_pass(teacher_dir: Path, target_dir: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(teacher_dir))
    if targets_ready(target_dir):
        print(f"Teacher logits already at {target_dir}", flush=True)
        return tokenizer
    from mlx_vlm import load

    print(f"Loading DWQ teacher {teacher_dir}", flush=True)
    teacher_vlm, _processor = load(str(teacher_dir))
    teacher = _TextLogits(teacher_vlm)
    train_data, valid_data = _load_calibration(tokenizer)
    from mlx_lm.quant.dwq import compute_dwq_targets

    print(f"Writing teacher logits to {target_dir}", flush=True)
    compute_dwq_targets(
        teacher,
        target_dir,
        train_data,
        valid_data,
        batch_size=DWQ_BATCH_SIZE,
        max_seq_length=DWQ_MAX_SEQ_LENGTH,
        seed=DWQ_SEED,
    )
    (target_dir / "settings.json").write_text(
        json.dumps(
            {
                "batch_size": DWQ_BATCH_SIZE,
                "max_seq_length": DWQ_MAX_SEQ_LENGTH,
                "seed": DWQ_SEED,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del teacher, teacher_vlm, _processor
    return tokenizer


def _student_pass(nn, teacher_dir: Path):
    from mlx_vlm import load

    print(f"Loading DWQ student {teacher_dir}", flush=True)
    student_vlm, _processor = load(str(teacher_dir))
    _quantize_all_linears(nn, student_vlm)
    safe_freeze(student_vlm)
    return student_vlm


def _watch_ceiling(stop: threading.Event) -> None:
    from playground.probe import MEMORY_CEILING_BYTES, enforce_metal_ceiling, phys_footprint

    while not stop.wait(0.5):
        enforce_metal_ceiling()
        if phys_footprint() >= MEMORY_CEILING_BYTES:
            print("DWQ crossed the 200GB ceiling. Stopping.", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)


def _release_mlx() -> None:
    import mlx.core as mx

    gc.collect()
    clear = getattr(mx, "clear_cache", None)
    if clear is not None:
        clear()
        return
    metal = getattr(mx, "metal", None)
    if metal is not None and hasattr(metal, "clear_cache"):
        metal.clear_cache()


def _checkpoint_layers(layers) -> None:
    from mlx_lm.tuner.trainer import grad_checkpoint

    for layer in layers:
        grad_checkpoint(layer)


def disk_target(target_dir: Path):
    """Load precomputed teacher logits. The teacher model is not in memory."""

    def target_fn(batch, index, split):
        import mlx.core as mx

        del batch
        saved = mx.load(str(target_dir / split / f"{index:010d}.safetensors"))
        return saved["logits"], saved["indices"]

    return target_fn


def freeze_module(module) -> None:
    """Freeze one module. Skip modules MLX cannot freeze.

    MLX stores parameters on the module itself, not in ``_parameters``.
    ``freeze(recurse=False)`` marks this module only, so a broken child
    such as ``AudioRelativePositionEmbedding`` can be skipped on its own.
    """
    freeze = getattr(module, "freeze", None)
    if freeze is None:
        return
    try:
        freeze(recurse=False)
    except AttributeError:
        return


def refuse_full_finetune(model) -> None:
    """Stop if DWQ is about to differentiate the full weight matrices."""
    from mlx.utils import tree_flatten
    from mlx_lm.utils import get_total_parameters

    total = get_total_parameters(model)
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    if total <= 0:
        raise SystemExit("DWQ student has no parameters.")
    fraction = trainable / total
    print(
        f"DWQ will train {fraction:.2%} of the weights ({trainable / 1e6:.1f}M/{total / 1e6:.1f}M).",
        flush=True,
    )
    if fraction > 0.15:
        raise SystemExit(
            "DWQ is about to train full weight matrices. That does not fit in 200GB, so it was stopped."
        )


def safe_freeze(model) -> None:
    model.apply_to_modules(lambda _prefix, module: freeze_module(module))


def teacher_target(teacher):
    """mlx calls ``target_fn(batch, index, split=...)``."""

    def target_fn(batch, index, split):
        del index, split
        return teacher(batch)

    return target_fn


def _quantize_all_linears(nn, model) -> None:
    import inspect

    def predicate(_path, module):
        return isinstance(module, nn.Linear)

    kwargs = {"group_size": DWQ_GROUP_SIZE, "bits": DWQ_BITS}
    if "class_predicate" in inspect.signature(nn.quantize).parameters:
        kwargs["class_predicate"] = predicate
    nn.quantize(model, **kwargs)


def distill_scales(model, target_fn, train_data, valid_data) -> tuple[float, float, int]:
    """Train 4-bit scales and restore the lowest validation point, including step 0.

    The loss matches mlx-lm 0.31.3 ``dwq_quantize``: temperature-scaled KL against
    the teacher's saved top-1024 logits. A later step that is worse is not saved.
    """
    import time

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optimizers
    from mlx.utils import tree_map
    from mlx_lm.tuner.losses import kl_div_loss
    from mlx_lm.tuner.trainer import iterate_batches
    from mlx_lm.tuner.utils import print_trainable_parameters
    from tqdm import tqdm

    def unfreeze(_prefix, module):
        if (
            hasattr(module, "bits")
            and hasattr(module, "group_size")
            and getattr(module, "mode", None) == "affine"
            and module.bits < 8
        ):
            module.unfreeze(keys=["scales", "biases"], recurse=False)

    model.train()
    model.apply_to_modules(unfreeze)
    print_trainable_parameters(model)
    refuse_full_finetune(model)

    group = mx.distributed.init()
    world_size = group.size()
    scale = 1 / DWQ_TEMPERATURE
    opt = optimizers.Adam(learning_rate=DWQ_LEARNING_RATE, bias_correction=True)

    def loss_fn(params, tokens, targets, lengths):
        model.update(tree_map(lambda value: value.astype(mx.bfloat16), params))
        logits = model(tokens)
        if isinstance(targets, tuple):
            targets, ids = targets
            logits = mx.take_along_axis(logits, ids, axis=-1)
        losses = kl_div_loss(scale * logits, scale * targets)
        mask = mx.arange(1, 1 + targets.shape[1]) < lengths[:, 1:]
        ntoks = mask.sum()
        loss = (mask * losses).sum() / ntoks
        return loss, ntoks

    def step(tokens, targets, lengths, params):
        (loss, ntoks), grads = mx.value_and_grad(loss_fn)(params, tokens, targets, lengths)
        grads = nn.average_gradients(grads)
        params = opt.apply_gradients(grads, params)
        return loss, ntoks, params

    def validate(params, step_index):
        total_loss = 0.0
        total_tokens = 0
        batches = iterate_batches(valid_data, DWQ_BATCH_SIZE, DWQ_MAX_SEQ_LENGTH, seed=DWQ_SEED)
        for index, (batch, lengths) in tqdm(
            enumerate(batches),
            total=len(valid_data) // DWQ_BATCH_SIZE,
            desc="Computing validation loss",
            leave=False,
        ):
            batch = batch[:, :-1]
            targets = target_fn(batch, index, split="valid")
            mx.eval(targets)
            loss, ntoks = loss_fn(params, batch, targets, lengths)
            mx.eval(loss, ntoks)
            loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
            ntoks = mx.distributed.all_sum(ntoks, stream=mx.cpu).item()
            total_tokens += ntoks
            total_loss += loss * ntoks
        mean = total_loss / total_tokens
        tqdm.write(f"Validation: it={step_index}, loss={mean:.3f}")
        return mean

    def snapshot(params):
        copied = tree_map(lambda value: mx.array(value), params)
        mx.eval(copied)
        return copied

    params = tree_map(lambda value: value.astype(mx.float32), model.trainable_parameters())
    tracked: tuple[int, float] | None = None
    best_params = None

    def note(step_index, loss, current):
        nonlocal tracked, best_params
        chosen = lower_validation(tracked, step_index, loss)
        if chosen != tracked:
            tracked = chosen
            best_params = snapshot(current)
            tqdm.write(f"Best validation so far: it={step_index}, loss={loss:.3f}")

    initial = validate(params, 0)
    note(0, initial, params)

    total_loss = 0.0
    total_tokens = 0
    window_tokens = 0
    started = time.time()
    last_index = 0
    batches = iterate_batches(train_data, DWQ_BATCH_SIZE, DWQ_MAX_SEQ_LENGTH, seed=DWQ_SEED)
    progress = tqdm(enumerate(batches), total=len(train_data) // DWQ_BATCH_SIZE)
    for it, (batch, lengths) in progress:
        last_index = it
        batch = batch[:, :-1]
        targets = target_fn(batch, it, split="train")
        mx.eval(targets)
        loss, ntoks, params = step(batch, targets, lengths, params)
        mx.eval(loss, params)
        loss = mx.distributed.all_sum(loss, stream=mx.cpu).item() / world_size
        ntoks = mx.distributed.all_sum(ntoks, stream=mx.cpu).item()
        window_tokens += ntoks
        total_loss += loss * ntoks
        progress.set_description(desc=f"{loss=:.4f}")
        if (it + 1) % 20 == 0:
            elapsed = time.time() - started
            avg_loss = total_loss / window_tokens
            total_tokens += window_tokens
            tqdm.write(
                f"it={it}, avg_loss={avg_loss:.4f}, total_tokens={total_tokens}, "
                f"toks_per_sec={window_tokens / elapsed:.3f}, "
                f"peak_memory_gb={mx.get_peak_memory() / 1e9:.3f}"
            )
            started = time.time()
            window_tokens = 0
            total_loss = 0.0
        if (it + 1) % 200 == 0:
            note(it, validate(params, it), params)

    final = validate(params, last_index)
    note(last_index, final, params)
    if tracked is None or best_params is None:
        raise SystemExit("DWQ did not record a validation loss.")
    best_step, best_loss = tracked
    model.update(tree_map(lambda value: value.astype(mx.bfloat16), best_params))
    tqdm.write(f"Restoring the best validation point it={best_step}, loss={best_loss:.3f}")
    return initial, best_loss, best_step


def _load_calibration(tokenizer):
    import mlx.core as mx
    import numpy as np
    from mlx_lm.quant.dwq import load_data

    np.random.seed(DWQ_SEED)
    mx.random.seed(DWQ_SEED)
    return load_data(
        tokenizer,
        DWQ_DATASET,
        DWQ_NUM_SAMPLES,
        DWQ_MAX_SEQ_LENGTH,
        num_valid_samples=DWQ_VALID_SAMPLES,
    )


class _TextLogits:
    """A token batch in, logits out, with the quantized VLM's parameters underneath."""

    def __init__(self, vlm):
        self.vlm = vlm
        language = getattr(vlm, "language_model", vlm)
        layers = getattr(language, "layers", None)
        if layers is None:
            inner = getattr(language, "model", None)
            layers = getattr(inner, "layers", None)
        if not layers:
            raise SystemExit("Gemma language model has no layers to checkpoint.")
        self.layers = layers

    def __call__(self, token_ids):
        return _forward_text(self.vlm, token_ids)

    def __getattr__(self, name):
        return getattr(self.vlm, name)


def _forward_text(model, token_ids):
    language = getattr(model, "language_model", None)
    target = language if language is not None else model
    try:
        out = target(token_ids)
    except TypeError:
        out = target(input_ids=token_ids)
    if isinstance(out, tuple):
        out = out[0]
    if hasattr(out, "logits"):
        out = out.logits
    return out


def _save_quantized(
    model,
    source: Path,
    dest: Path,
    initial: float,
    best: float,
    best_step: int,
) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in source.iterdir():
        if not item.is_file():
            continue
        if item.suffix == ".safetensors" or item.name == "model.safetensors.index.json":
            continue
        shutil.copy2(item, dest / item.name)
    config_path = dest / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization"] = {
        "bits": DWQ_BITS,
        "group_size": DWQ_GROUP_SIZE,
        "mode": "affine",
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    model.save_weights(str(dest / "model.safetensors"))
    report = {
        "bits": DWQ_BITS,
        "group_size": DWQ_GROUP_SIZE,
        "num_samples": DWQ_NUM_SAMPLES,
        "num_valid_samples": DWQ_VALID_SAMPLES,
        "max_seq_length": DWQ_MAX_SEQ_LENGTH,
        "temperature": DWQ_TEMPERATURE,
        "learning_rate": DWQ_LEARNING_RATE,
        "batch_size": DWQ_BATCH_SIZE,
        "dataset": DWQ_DATASET,
        "initial_loss": initial,
        "best_loss": best,
        "best_step": best_step,
        "final_loss": best,
    }
    (dest / "dwq.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
