import pytest

from playground.train import TRAIN_CONTEXT, build_train_set, validate_train_args


def _count(text: str) -> int:
    return len(text)


def _row(split: str, page: str, commission: str = "What does this file do?") -> dict:
    return {
        "id": split + page[:8],
        "split": split,
        "commission": commission,
        "report": "It returns 1.",
        "thinking": [{"round": 0, "text": "secret thought"}],
        "tool_rounds": [
            {
                "name": "read_file",
                "arguments": {"path": "a.py", "start_line": 1, "end_line": 2},
                "result": {"content": page},
            }
        ],
    }


def test_holdout_rows_are_excluded_and_pages_are_trimmed():
    train = _row("train", "KEEP_PAGE" + ("x" * 400))
    holdout = _row("holdout", "HOLDOUT_PAGE")
    packed, skipped = build_train_set([train, holdout], 250, _count)
    assert skipped == 1
    assert len(packed) == 1
    assert "HOLDOUT_PAGE" not in packed[0]["text"]
    assert "secret thought" not in packed[0]["text"]
    assert packed[0]["tokens"] <= 250
    assert "x" * 400 not in packed[0]["text"]
    assert " " * 20 not in packed[0]["text"]


def test_overlong_skeleton_is_skipped():
    huge = _row("train", "PAGE", commission="Q" * 500)
    small = _row("train", "KEEP")
    with pytest.raises(SystemExit):
        build_train_set([huge], 40, _count)
    packed, skipped = build_train_set([huge, small], 400, _count)
    assert skipped == 1
    assert len(packed) == 1
    assert "KEEP" in packed[0]["text"]


def test_batch_size_above_one_is_rejected():
    with pytest.raises(SystemExit):
        validate_train_args(TRAIN_CONTEXT, 2)


def test_ceiling_is_200gb():
    from playground.probe import MEMORY_CEILING_BYTES

    assert MEMORY_CEILING_BYTES == 200 * 1024 ** 3


def test_context_above_32k_is_rejected():
    with pytest.raises(SystemExit):
        validate_train_args(98304, 1)
