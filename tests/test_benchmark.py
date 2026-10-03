import json
from pathlib import Path

from playground.benchmark import check_task, compare_reports, load_tasks, report_path


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_string_line_numbers_are_valid_read_args():
    from playground.score import _args_ok

    assert _args_ok("read_file", {"path": "a.py", "start_line": "1", "end_line": "4"}) is True
    tasks = load_tasks()
    kinds = {task["kind"] for task in tasks}
    assert kinds == {"locate", "read-adjacent", "paraphrase", "negative"}
    assert len(tasks) == 8
    assert all(task["id"] and task["objective"] and task["gold"] for task in tasks)


def test_locate_passes_when_grep_hits_and_report_names_the_file(tmp_path: Path):
    rel = "src/scout/code/commissions.py"
    _write(tmp_path, rel, "def research_junior_tools():\n    return []\n")
    task = {
        "id": "locate",
        "kind": "locate",
        "checks": {
            "path": rel,
            "symbol": "research_junior_tools",
            "forbid_absence_claim": True,
        },
    }
    flags = check_task(
        task,
        {
            "report": "research_junior_tools is defined in src/scout/code/commissions.py.",
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "grep",
                    "arguments": {"pattern": "research_junior_tools"},
                    "result": {
                        "matches": [{"path": rel, "line": 1, "text": "def research_junior_tools():"}]
                    },
                }
            ],
        },
        tmp_path,
    )
    assert flags["passed"] is True


def test_read_adjacent_fails_without_opening_the_file(tmp_path: Path):
    rel = "src/scout/oa/runner.py"
    quote = "Stop exploring. Write the report now from what you already have."
    _write(tmp_path, rel, f'RESEARCH_WRAPUP = (\n    "{quote}"\n)\n')
    task = {
        "id": "wrap",
        "kind": "read-adjacent",
        "checks": {
            "path": rel,
            "symbol": "RESEARCH_WRAPUP",
            "must_read": True,
            "quote": quote,
            "forbid_absence_claim": True,
        },
    }
    flags = check_task(
        task,
        {
            "report": f"RESEARCH_WRAPUP says {quote}",
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "grep",
                    "arguments": {"pattern": "RESEARCH_WRAPUP"},
                    "result": {"matches": [{"path": rel, "line": 1, "text": "RESEARCH_WRAPUP = ("}]},
                }
            ],
        },
        tmp_path,
    )
    assert flags["read_file"] is False
    assert flags["passed"] is False


def test_read_adjacent_passes_when_the_page_and_report_quote_it(tmp_path: Path):
    rel = "src/scout/oa/runner.py"
    quote = "Stop exploring. Write the report now from what you already have."
    _write(tmp_path, rel, f"{quote}\n")
    task = {
        "id": "wrap",
        "kind": "read-adjacent",
        "checks": {
            "path": rel,
            "symbol": "RESEARCH_WRAPUP",
            "must_read": True,
            "quote": quote,
            "forbid_absence_claim": True,
        },
    }
    flags = check_task(
        task,
        {
            "report": f"RESEARCH_WRAPUP in src/scout/oa/runner.py: {quote}",
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "read_file",
                    "arguments": {"path": rel, "start_line": 1, "end_line": 5},
                    "result": {"path": rel, "content": quote + "\n"},
                }
            ],
        },
        tmp_path,
    )
    assert flags["passed"] is True


def test_negative_fails_on_empty_grep(tmp_path: Path):
    task = {
        "id": "missing",
        "kind": "negative",
        "checks": {"symbol": "definitely_not_in_uce_reader_xyz", "expect_absent": True},
    }
    flags = check_task(
        task,
        {
            "report": "I cannot locate definitely_not_in_uce_reader_xyz.",
            "tool_rounds": [{"round": 0, "name": "grep", "arguments": {}, "result": {"error": "missing_pattern"}}],
        },
        tmp_path,
    )
    assert flags["valid_tools"] is False
    assert flags["passed"] is False


def test_string_line_numbers_are_valid_read_args():
    from playground.score import _args_ok

    assert _args_ok("read_file", {"path": "a.py", "start_line": "1", "end_line": "4"}) is True


def test_not_found_counts_as_an_absence_claim(tmp_path: Path):
    task = {
        "id": "missing",
        "kind": "negative",
        "checks": {"symbol": "definitely_not_in_uce_reader_xyz", "expect_absent": True},
    }
    flags = check_task(
        task,
        {
            "report": "The function was not found.",
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "grep",
                    "arguments": {"pattern": "definitely_not_in_uce_reader_xyz"},
                    "result": {"matches": [], "count": 0},
                }
            ],
        },
        tmp_path,
    )
    assert flags["claimed_absent"] is True
    assert flags["passed"] is True


def test_negative_passes_when_the_report_says_it_is_absent(tmp_path: Path):
    task = {
        "id": "missing",
        "kind": "negative",
        "checks": {"symbol": "definitely_not_in_uce_reader_xyz", "expect_absent": True},
    }
    flags = check_task(
        task,
        {
            "report": "definitely_not_in_uce_reader_xyz does not exist in this repo.",
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "grep",
                    "arguments": {"pattern": "definitely_not_in_uce_reader_xyz"},
                    "result": {"matches": [], "count": 0},
                }
            ],
        },
        tmp_path,
    )
    assert flags["passed"] is True


def test_false_absence_fails_a_locate(tmp_path: Path):
    rel = "src/scout/code/prompts.py"
    _write(tmp_path, rel, "RESEARCH_SKILL = 'report'\n")
    task = {
        "id": "prompt",
        "kind": "paraphrase",
        "checks": {"path": rel, "forbid_absence_claim": True},
    }
    flags = check_task(
        task,
        {
            "report": "I cannot locate the research prompt in this repo.",
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "grep",
                    "arguments": {"pattern": "research junior"},
                    "result": {"matches": [{"path": rel, "line": 1, "text": "RESEARCH_SKILL"}]},
                }
            ],
        },
        tmp_path,
    )
    assert flags["no_false_absence"] is False
    assert flags["passed"] is False


def test_missing_target_file_is_invalid(tmp_path: Path):
    flags = check_task(
        {"checks": {"path": "missing.py", "symbol": "nope"}},
        {"report": "missing.py defines nope", "tool_rounds": []},
        tmp_path,
    )
    assert flags["invalid"] is True
    assert flags["passed"] is False


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
