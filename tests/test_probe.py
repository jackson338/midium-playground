import json
from pathlib import Path

import pytest

from playground.probe import (
    MEMORY_CEILING_BYTES,
    assert_previous_probe,
    build_messages,
    pack_example,
    render_messages,
    select_train_episode,
)


def _count(text: str) -> int:
    return len(text)


def _episode(pages: list[str]) -> dict:
    rounds = []
    for index, page in enumerate(pages):
        rounds.append(
            {
                "name": "read_file",
                "arguments": {"path": f"f{index}.py", "start_line": 1, "end_line": 2},
                "result": {"content": page},
            }
        )
    return {
        "id": "ep",
        "split": "train",
        "token_estimate": 100,
        "commission": "What does the file do?",
        "report": "It returns 1.",
        "thinking": [{"round": 0, "text": "secret thought"}],
        "tool_rounds": rounds,
    }


def test_trim_drops_later_pages_and_does_not_pad():
    episode = _episode(["AAA_PAGE", "BBB_PAGE", "CCC_PAGE"])
    full = render_messages(build_messages(episode, ["AAA_PAGE", "BBB_PAGE", "CCC_PAGE"]))
    packed = pack_example(episode, len(full) - 5, _count)
    assert packed is not None
    assert "CCC_PAGE" not in packed["text"]
    assert "AAA_PAGE" in packed["text"]
    assert packed["text"].strip() == packed["text"].strip()
    assert " " * 20 not in packed["text"]
    assert "secret thought" not in packed["text"]


def test_96k_pack_uses_real_read_file_text():
    page = "REAL_PAGE_TEXT_" + ("x" * 200)
    episode = _episode([page, "SECOND_PAGE"])
    packed = pack_example(episode, 98304, _count)
    assert packed is not None
    assert "REAL_PAGE_TEXT_" in packed["text"]
    assert packed["pages_used"] >= 1


def test_96k_pack_drops_when_skeleton_does_not_fit():
    episode = _episode(["page"])
    episode["commission"] = "Q" * 50
    packed = pack_example(episode, 98304, lambda text: 200_000)
    assert packed is None


def test_gate_refuses_when_previous_probe_was_killed(tmp_path: Path):
    previous = tmp_path / "probe-16k.json"
    previous.write_text(
        json.dumps({"status": "killed", "peak_bytes": 10, "context": 16384}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        assert_previous_probe(32768, tmp_path)


def test_gate_refuses_when_peak_crosses_the_ceiling(tmp_path: Path):
    previous = tmp_path / "probe-32k.json"
    previous.write_text(
        json.dumps({"status": "finished", "peak_bytes": MEMORY_CEILING_BYTES, "context": 32768}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        assert_previous_probe(98304, tmp_path)


def test_selects_the_longest_train_episode_with_a_page():
    short = _episode(["a"])
    short["id"] = "a"
    short["token_estimate"] = 10
    long = _episode(["abcdef"])
    long["id"] = "b"
    long["token_estimate"] = 90
    holdout = _episode(["zzzz"])
    holdout["split"] = "holdout"
    holdout["token_estimate"] = 500
    chosen = select_train_episode([short, holdout, long])
    assert chosen["id"] == "b"
