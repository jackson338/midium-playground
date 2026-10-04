import json
from pathlib import Path

from playground.benchmark import compare_reports, report_path


def test_string_line_numbers_are_valid_read_args():
    from playground.score import _args_ok

    assert _args_ok("read_file", {"path": "a.py", "start_line": "1", "end_line": "4"}) is True


def test_compare_prints_a_task_that_moved(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("playground.benchmark.BENCHMARKS_DIR", tmp_path)
    before = {
        "pass_rate": 0.0,
        "good_rate": 0.0,
        "tasks": [{"id": "wrap", "deterministic": {"passed": False}, "semantic": {"grade": "bad"}}],
    }
    after = {
        "pass_rate": 1.0,
        "good_rate": 1.0,
        "tasks": [{"id": "wrap", "deterministic": {"passed": True}, "semantic": {"grade": "good"}}],
    }
    report_path("before").write_text(json.dumps(before), encoding="utf-8")
    report_path("after").write_text(json.dumps(after), encoding="utf-8")
    text = compare_reports("before", "after")
    assert "check False -> True" in text
    assert "grade bad -> good" in text
