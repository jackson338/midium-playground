import json
from pathlib import Path

import pytest

from playground import qwen_flash


def _ready(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    (path / "model.safetensors.index.json").write_text("{}")


def test_stale_ple_view_drops_ngram_scales(tmp_path):
    view = tmp_path / "qwen-ple"
    view.mkdir()
    (view / "ple-store.json").write_text("{}\n")
    (view / "config.json").write_text("{}\n")
    scale = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    index = {
        "weight_map": {
            scale: "model-00001-of-00028.safetensors",
            "language_model.model.embed_tokens.weight": "model-00002-of-00028.safetensors",
        }
    }
    (view / "model.safetensors.index.json").write_text(json.dumps(index))
    assert qwen_flash.ple_ready(view) is False
    qwen_flash.drop_resident_ple_keys(view)
    kept = json.loads((view / "model.safetensors.index.json").read_text())["weight_map"]
    assert scale not in kept
    assert kept == {
        "language_model.model.embed_tokens.weight": "model-00002-of-00028.safetensors"
    }
    assert qwen_flash.ple_ready(view) is True


def test_unpacked_row_width_uses_bits():
    assert qwen_flash.unpacked_row_width(30, 6) == 160
    assert qwen_flash.unpacked_row_width(20, 4) == 160
    with pytest.raises(ValueError):
        qwen_flash.unpacked_row_width(1, 6)


def test_download_runs_only_when_the_dir_is_incomplete(tmp_path, monkeypatch):
    model = tmp_path / "qwen"
    monkeypatch.setattr(qwen_flash, "MODEL_DIR", model)
    calls = []

    def fake_download(repo, local_dir):
        calls.append(repo)
        _ready(Path(local_dir))

    monkeypatch.setattr(qwen_flash, "snapshot_download", fake_download)
    first = qwen_flash.ensure_weights()
    second = qwen_flash.ensure_weights()
    assert first == model
    assert second == model
    assert calls == [qwen_flash.HF_REPO]


def test_prints_the_completion(tmp_path, monkeypatch, capsys):
    model = tmp_path / "qwen"
    _ready(model)
    monkeypatch.setattr(qwen_flash, "MODEL_DIR", model)

    def fail_download(repo, local_dir):
        raise AssertionError(repo)

    monkeypatch.setattr(qwen_flash, "snapshot_download", fail_download)
    monkeypatch.setattr(qwen_flash, "ensure_ple_view", lambda source: source)
    monkeypatch.setattr(qwen_flash, "negotiate_metal_memory", lambda: None)
    monkeypatch.setattr(qwen_flash, "load_lenient", lambda path: ("model", "processor"))
    monkeypatch.setattr(
        qwen_flash,
        "complete",
        lambda model, processor, prompt, max_tokens, model_dir: "def fib(n):\n    return n",
    )
    qwen_flash.run_qwen_cli("Write a Python Fibonacci function.", max_tokens=32)
    assert capsys.readouterr().out == "def fib(n):\n    return n\n"
