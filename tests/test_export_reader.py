import json

import pytest

import inspect

from playground.export_reader import (
    BASE_COMMIT,
    BASE_MODEL,
    check_adapter,
)
import playground.export_reader as export_reader
from playground.compare_lora import print_scores
from playground.subswe import F16_MODEL, LORA_MODEL, READER_4BIT_MODEL, trace_path


def _adapter(tmp_path, base: str = BASE_MODEL, commit: str = BASE_COMMIT):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "adapters.safetensors").write_bytes(b"weights")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base,
                "base_model_commit_hash": commit,
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_cached_main_snapshot_is_used_when_the_commit_dir_is_absent(tmp_path):
    from playground.export_reader import local_gemma_snapshot

    repo = tmp_path / "models--unsloth--gemma-4-E4B-it"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc123\n", encoding="utf-8")
    assert local_gemma_snapshot(tmp_path) == str(snapshot)


def test_missing_cache_is_refused(tmp_path):
    from playground.export_reader import local_gemma_snapshot

    with pytest.raises(SystemExit, match="not in the local Hugging Face cache"):
        local_gemma_snapshot(tmp_path)


def test_missing_adapter_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        check_adapter(tmp_path / "missing")


def test_wrong_base_is_refused(tmp_path):
    path = _adapter(tmp_path / "adapter", base="google/gemma-4-e4b-it")
    with pytest.raises(SystemExit, match="Adapter base"):
        check_adapter(path)


def test_fuse_does_not_call_mlx_vlm_fuse():
    assert "mlx_vlm.fuse" not in inspect.getsource(export_reader)


def test_dwq_settings_are_the_high_quality_recipe():
    from playground.dwq_vlm import (
        DWQ_BATCH_SIZE,
        DWQ_BITS,
        DWQ_GROUP_SIZE,
        DWQ_LEARNING_RATE,
        DWQ_MAX_SEQ_LENGTH,
        DWQ_NUM_SAMPLES,
        DWQ_TEMPERATURE,
        DWQ_VALID_SAMPLES,
    )

    assert DWQ_BITS == 4
    assert DWQ_GROUP_SIZE == 32
    assert DWQ_NUM_SAMPLES == 1024
    assert DWQ_VALID_SAMPLES == 32
    assert DWQ_MAX_SEQ_LENGTH == 512
    assert DWQ_TEMPERATURE == 2.0
    assert DWQ_LEARNING_RATE == 1e-5
    assert DWQ_BATCH_SIZE == 1
    assert "mlx_lm.dwq" not in inspect.getsource(export_reader)


def test_freeze_skips_a_module_without_no_grad():
    from playground.dwq_vlm import freeze_module

    class _Broken:
        def freeze(self, *, recurse=True):
            raise AttributeError("_no_grad")

    class _Ok:
        def __init__(self):
            self.recurse = None

        def freeze(self, *, recurse=True):
            self.recurse = recurse

    freeze_module(_Broken())
    ok = _Ok()
    freeze_module(ok)
    assert ok.recurse is False


def test_teacher_logits_are_reused_only_when_both_splits_exist(tmp_path):
    from playground.dwq_vlm import DWQ_BATCH_SIZE, DWQ_MAX_SEQ_LENGTH, DWQ_SEED, targets_ready

    assert targets_ready(tmp_path) is False
    (tmp_path / "train").mkdir()
    (tmp_path / "valid").mkdir()
    (tmp_path / "train" / "0000000000.safetensors").write_bytes(b"x")
    (tmp_path / "valid" / "0000000000.safetensors").write_bytes(b"x")
    (tmp_path / "settings.json").write_text(
        json.dumps({"batch_size": DWQ_BATCH_SIZE, "max_seq_length": DWQ_MAX_SEQ_LENGTH}),
        encoding="utf-8",
    )
    assert targets_ready(tmp_path) is False
    (tmp_path / "settings.json").write_text(
        json.dumps(
            {
                "batch_size": DWQ_BATCH_SIZE,
                "max_seq_length": DWQ_MAX_SEQ_LENGTH,
                "seed": DWQ_SEED,
            }
        ),
        encoding="utf-8",
    )
    assert targets_ready(tmp_path) is True


def test_teacher_target_accepts_the_split_keyword():
    from playground.dwq_vlm import teacher_target

    seen = teacher_target(lambda batch: batch)("tokens", 0, split="valid")
    assert seen == "tokens"


def test_worse_validation_loss_refuses_the_benchmark():
    from playground.dwq_vlm import accept_distillation, validation_losses

    initial, final = validation_losses(
        ["Validation: it=0, loss=1.200", "Validation: it=10, loss=0.800"]
    )
    assert (initial, final) == (1.2, 0.8)
    accept_distillation(initial, final)
    with pytest.raises(SystemExit, match="worse"):
        accept_distillation(1.2, 1.4)


def test_finished_f16_directory_skips_the_fuse(tmp_path):
    from playground.export_reader import export_steps

    bf16 = tmp_path / "bf16"
    four = tmp_path / "four"
    bf16.mkdir()
    (bf16 / "config.json").write_text("{}", encoding="utf-8")
    (bf16 / "model.safetensors").write_bytes(b"weights")
    assert export_steps(bf16, four) == ["dwq", "score"]
    (four).mkdir()
    (four / "config.json").write_text('{"quantization": {"bits": 4}}', encoding="utf-8")
    (four / "model.safetensors").write_bytes(b"weights")
    assert export_steps(bf16, four) == ["score"]


def test_head_to_head_lists_three_pass_rates(capsys):
    def row(model: str, rate: float, run: int) -> dict:
        return {
            "model": model,
            "run": run,
            "pass_rate": rate,
            "passed": 1,
            "attempted": 40,
            "call_count": 10,
            "by_kind": {"locate": {"passed": 1, "attempted": 6}},
            "tasks": [],
        }

    print_scores(
        [
            row(F16_MODEL, 0.65, 1),
            row(LORA_MODEL, 0.775, 3),
            row(READER_4BIT_MODEL, 0.75, 1),
        ]
    )
    out = capsys.readouterr().out
    assert f"{F16_MODEL}: pass_rate=0.65" in out
    assert f"{LORA_MODEL}: pass_rate=0.775" in out
    assert f"{READER_4BIT_MODEL}: pass_rate=0.75" in out
    assert "average" not in out.lower()
    assert trace_path(READER_4BIT_MODEL, 1).name == "gemma-4-e4b-reader-4bit.run1.jsonl"
