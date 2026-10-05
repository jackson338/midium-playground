import json

import pytest

import inspect

from playground.export_reader import (
    BASE_COMMIT,
    BASE_MODEL,
    check_adapter,
    dwq_command,
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


def test_dwq_uses_the_fused_f16_folder(tmp_path):
    bf16 = tmp_path / "bf16"
    four = tmp_path / "four"
    dwq = dwq_command(bf16, four)
    assert dwq[2] == "mlx_lm.dwq"
    assert "--bits" in dwq and "4" in dwq
    assert str(bf16) in dwq
    assert str(four) in dwq


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
