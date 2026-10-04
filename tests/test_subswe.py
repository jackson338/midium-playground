import json
from pathlib import Path

from playground.benchmark import check_task, load_tasks
from playground.subswe import load_subswe_tasks


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _episode(report: str, rounds: list[dict]) -> dict:
    return {"report": report, "stop_reason": "report", "tool_rounds": rounds}


def test_uce_smoke_is_separate_from_subswe():
    smoke = load_tasks()
    public = load_subswe_tasks()
    assert len(smoke) == 8
    assert all(task.get("private") for task in smoke)
    assert len(public) == 40
    assert {task["kind"] for task in public} == {
        "locate",
        "read",
        "paraphrase",
        "negative",
        "multi-file",
        "long-page",
        "distractor",
    }


def test_string_line_numbers_fail_valid_tools(tmp_path: Path):
    rel = "a.py"
    _write(tmp_path, rel, "class Widget:\n    pass\n")
    flags = check_task(
        {"kind": "read", "gold": {"paths": [rel], "symbol": "Widget", "must_read": True, "quote": "class Widget:"}},
        _episode(
            "class Widget: is in a.py",
            [
                {
                    "name": "read_file",
                    "arguments": {"path": rel, "start_line": "1", "end_line": "2"},
                    "result": {"path": rel, "content": "class Widget:\n    pass\n"},
                }
            ],
        ),
        tmp_path,
    )
    assert flags["valid_tools"] is False
    assert flags["passed"] is False


def test_sixteen_calls_still_stop_clean(tmp_path: Path):
    rel = "a.py"
    _write(tmp_path, rel, "class Widget:\n")
    rounds = [
        {"name": "grep", "arguments": {"pattern": "Widget"}, "result": {"matches": []}}
        for _ in range(16)
    ]
    flags = check_task(
        {"kind": "locate", "gold": {"paths": [rel], "symbol": "Widget"}},
        _episode("Widget is in a.py", rounds),
        tmp_path,
    )
    assert flags["stopped_clean"] is True
    flags = check_task(
        {"kind": "locate", "gold": {"paths": [rel], "symbol": "Widget"}},
        _episode("Widget is in a.py", rounds + rounds[:1]),
        tmp_path,
    )
    assert flags["stopped_clean"] is False


def test_tool_call_leak_fails_stopped_clean(tmp_path: Path):
    rel = "a.py"
    _write(tmp_path, rel, "class Widget:\n")
    flags = check_task(
        {"kind": "locate", "gold": {"paths": [rel], "symbol": "Widget"}},
        _episode(
            "<|tool_call>call:glob\nWidget is in a.py",
            [{"name": "grep", "arguments": {"pattern": "Widget"}, "result": {"matches": [{"path": rel}]}}],
        ),
        tmp_path,
    )
    assert flags["stopped_clean"] is False
    assert flags["passed"] is False


def test_negative_glob_is_not_a_search(tmp_path: Path):
    task = {
        "kind": "negative",
        "gold": {"symbol": "definitely_missing_xyz", "expect_absent": True},
    }
    flags = check_task(
        task,
        _episode(
            "definitely_missing_xyz does not exist.",
            [{"name": "glob", "arguments": {"pattern": "**/*.py"}, "result": {"matches": []}}],
        ),
        tmp_path,
    )
    assert flags["negative_search"] is False
    assert flags["symbol_seen"] is None
    assert flags["named_paths"] is None
    assert flags["passed"] is False


def test_negative_grep_and_absence_passes(tmp_path: Path):
    flags = check_task(
        {"kind": "negative", "gold": {"symbol": "definitely_missing_xyz", "expect_absent": True}},
        _episode(
            "definitely_missing_xyz was not found.",
            [{"name": "grep", "arguments": {"pattern": "definitely_missing_xyz"}, "result": {"matches": []}}],
        ),
        tmp_path,
    )
    assert flags["negative_search"] is True
    assert flags["passed"] is True


def test_short_read_misses_a_quote_past_line_500(tmp_path: Path):
    body = "".join(f"line {i}\n" for i in range(1, 600)) + "LATE_QUOTE_MARKER\n"
    rel = "wide.py"
    _write(tmp_path, rel, body)
    early = "".join(f"line {i}\n" for i in range(1, 501))
    flags = check_task(
        {"kind": "long-page", "gold": {"paths": [rel], "must_read": True, "quote": "LATE_QUOTE_MARKER"}},
        _episode(
            "I read the start of wide.py.",
            [{"name": "read_file", "arguments": {"path": rel, "start_line": 1, "end_line": 500}, "result": {"path": rel, "content": early}}],
        ),
        tmp_path,
    )
    assert flags["read_page"] is False
    assert flags["passed"] is False


def test_cloud_error_is_excluded_from_the_rate(tmp_path: Path):
    from playground.subswe import _rescore_file

    task_id = "locate-one"
    tasks = {
        task_id: {
            "id": task_id,
            "kind": "locate",
            "path": str(tmp_path),
            "gold": {"paths": ["a.py"], "symbol": "Widget"},
        }
    }
    _write(tmp_path, "a.py", "class Widget:\n")
    dest = tmp_path / "model.run1.jsonl"
    dest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": task_id,
                        "error": "Midium Cloud 500: boom",
                        "tool_rounds": [],
                        "semantic": None,
                    }
                ),
                json.dumps(
                    {
                        "id": task_id,
                        "error": None,
                        "report": "Widget is in a.py",
                        "stop_reason": "report",
                        "tool_rounds": [
                            {"name": "grep", "arguments": {"pattern": "Widget"}, "result": {"matches": []}}
                        ],
                        "semantic": {"grade": None, "reason": "grader down", "judge_error": True},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Both rows share an id; the rescore keeps both. The error row is excluded.
    scored = _rescore_file(dest, tasks, "Gemma 4 E4B", 1)
    assert scored["error_count"] == 1
    assert scored["attempted"] == 1
    assert scored["pass_rate"] == 1.0
