"""Distill a 4-bit Gemma 4 VLM from a fused bf16 teacher.

mlx-lm's DWQ command only loads text models. This runs the same loss,
``dwq_quantize`` and ``kl_div_loss``, on the language, vision, and audio
towers of the fused VLM. Vision and audio are quantized. The calibration
text does not teach them.
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
DWQ_LEARNING_RATE = 1e-5
DWQ_BATCH_SIZE = 1
DWQ_DATASET = "allenai/tulu-3-sft-mixture"


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
    if saved.get("batch_size") != DWQ_BATCH_SIZE or saved.get("max_seq_length") != DWQ_MAX_SEQ_LENGTH:
        return False
    return any((path / "train").glob("*.safetensors")) and any((path / "valid").glob("*.safetensors"))


def run_vlm_dwq(teacher_dir: Path, dest: Path) -> None:
    """Quantize ``teacher_dir`` and save a 4-bit VLM at ``dest`` when the loss improves.

    The teacher and the student are never loaded together. Teacher logits are
    written to disk first, then the teacher is freed before the student trains.
    """
    import mlx.nn as nn
    import mlx.optimizers as optimizers
    from tqdm import tqdm

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
        _cap_wired_limit(enforce_metal_ceiling)
        lines: list[str] = []
        real_write = tqdm.write

        def _write(*args, **kwargs):
            lines.append(" ".join(str(arg) for arg in args))
            return real_write(*args, **kwargs)

        tqdm.write = _write
        try:
            from mlx_lm.quant.dwq import dwq_quantize

            _checkpoint_layers(student.layers)
            dwq_quantize(
                student,
                disk_target(target_dir),
                optimizers.Adam(learning_rate=DWQ_LEARNING_RATE, bias_correction=True),
                train_data,
                valid_data,
                batch_size=DWQ_BATCH_SIZE,
                max_seq_length=DWQ_MAX_SEQ_LENGTH,
                seed=123,
                temperature=DWQ_TEMPERATURE,
                gradient_checkpoint=False,
            )
        finally:
            tqdm.write = real_write
            enforce_metal_ceiling()
        initial, final = validation_losses(lines)
        accept_distillation(initial, final)
        _save_quantized(student_vlm, teacher_dir, dest, initial, final)
        print(f"DWQ 4-bit saved to {dest} (loss {initial:.4f} -> {final:.4f})", flush=True)
    finally:
        stop.set()


def _teacher_pass(teacher_dir: Path, target_dir: Path):
    from mlx_vlm import load

    if targets_ready(target_dir):
        print(f"Teacher logits already at {target_dir}", flush=True)
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(teacher_dir))
    print(f"Loading DWQ teacher {teacher_dir}", flush=True)
    teacher_vlm, processor = load(str(teacher_dir))
    teacher = _TextLogits(teacher_vlm)
    tokenizer = getattr(processor, "tokenizer", processor)
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
        seed=123,
    )
    (target_dir / "settings.json").write_text(
        json.dumps(
            {"batch_size": DWQ_BATCH_SIZE, "max_seq_length": DWQ_MAX_SEQ_LENGTH},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del teacher, teacher_vlm, processor
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
    """Mark one module's parameters frozen. Skip modules MLX cannot freeze."""
    try:
        parameters = object.__getattribute__(module, "_parameters")
        no_grad = object.__getattribute__(module, "_no_grad")
    except AttributeError:
        return
    no_grad.update(parameters.keys())


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


def _load_calibration(tokenizer):
    from mlx_lm.quant.dwq import load_data

    return load_data(
        tokenizer,
        DWQ_DATASET,
        DWQ_NUM_SAMPLES,
        DWQ_MAX_SEQ_LENGTH,
        num_valid_samples=DWQ_VALID_SAMPLES,
    )


def _cap_wired_limit(enforce_metal_ceiling) -> None:
    import mlx_lm.quant.dwq as dwq_mod

    def capped():
        from mlx_lm.utils import maybe_set_recommended_wired_limit

        try:
            maybe_set_recommended_wired_limit()
        finally:
            enforce_metal_ceiling()

    dwq_mod.maybe_set_recommended_wired_limit = capped


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


def _save_quantized(model, source: Path, dest: Path, initial: float, final: float) -> None:
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
        "final_loss": final,
    }
    (dest / "dwq.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
