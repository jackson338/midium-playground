import json
from pathlib import Path

from playground.config import MAX_CONTEXT_TOKENS
from playground.harness.shell import CommandClass, Workspace, classify_os_bash
from playground.harness.tools import ReaderTools
from playground.repos import is_holdout, search_query
from playground.score import score_episode
from playground.traces import build_episode, estimate_tokens


def test_read_file_pages_raw_lines(tmp_path: Path):
    file_path = tmp_path / "mod.py"
    file_path.write_text("".join(f"line {i}\n" for i in range(1, 21)), encoding="utf-8")
    tools = ReaderTools(Workspace(tmp_path))
    page = tools.read_file("mod.py", start_line=3, end_line=5)
    assert page["content"] == "line 3\nline 4\nline 5\n"
    assert page["truncated"] is False
    assert page["total_lines"] == 20
    assert "summary" not in page["content"].lower() or "line" in page["content"]


def test_read_file_truncates_with_next_start(tmp_path: Path):
    file_path = tmp_path / "wide.py"
    file_path.write_text("".join(f"row {i} {'x' * 80}\n" for i in range(1, 5001)), encoding="utf-8")
    tools = ReaderTools(Workspace(tmp_path))
    page = tools.read_file("wide.py", start_line=1, end_line=5000)
    assert page["truncated"] is True
    assert page["next_start_line"] == page["end_line"] + 1
    assert page["content"].startswith("row 1 ")
    assert "Purpose:" not in page["content"]


def test_read_file_rejects_paths_outside_workspace(tmp_path: Path):
    tools = ReaderTools(Workspace(tmp_path))
    result = tools.read_file("/etc/hosts", start_line=1, end_line=2)
    assert result["error"] == "outside_work_root"


def test_glob_finds_a_python_file(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    found = ReaderTools(Workspace(tmp_path)).glob("**/*.py")
    assert "error" not in found
    assert any(item["path"] == "a.py" for item in found["matches"])


def test_glob_returns_an_error_when_pathlib_rejects_the_pattern(tmp_path: Path, monkeypatch):
    class _Reject:
        def __iter__(self):
            raise ValueError("Invalid pattern: '**' can only be an entire path component")

    monkeypatch.setattr(Path, "glob", lambda self, pattern: _Reject())
    result = ReaderTools(Workspace(tmp_path)).dispatch("glob", {"pattern": "**.py"})
    assert result["error"] == "invalid_pattern"
    assert "entire path component" in result["detail"]


def test_bash_write_is_refused(tmp_path: Path):
    assert classify_os_bash("git commit -m x") == CommandClass.WRITE
    assert classify_os_bash("git status") == CommandClass.READ
    result = ReaderTools(Workspace(tmp_path)).os_bash("rm -rf .")
    assert result["exit_code"] == 1
    assert "not allowed" in result["output"]


def test_grep_finds_line(tmp_path: Path):
    (tmp_path / "a.py").write_text("def answer():\n    return 7\n", encoding="utf-8")
    found = ReaderTools(Workspace(tmp_path)).grep("def answer")
    assert found["count"] == 1
    assert found["matches"][0]["line"] == 1


def test_grep_file_path_searches_that_file(tmp_path: Path):
    (tmp_path / "addon.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (tmp_path / "other.py").write_text("def keep():\n    return 2\n", encoding="utf-8")
    found = ReaderTools(Workspace(tmp_path)).grep("def keep", path="addon.py")
    assert found.get("error") is None
    assert found["count"] == 1
    assert found["matches"][0]["path"].endswith("addon.py")
    assert "other.py" not in found["matches"][0]["path"]


def test_search_query_is_one_license():
    query = search_query("mit", "Python")
    assert "license:mit" in query
    assert " OR " not in query
    assert "(" not in query


def test_holdout_skips_product_trees():
    assert is_holdout("jacksonoaks/unified_compute_engine")
    assert is_holdout("acme/midium")
    assert not is_holdout("pallets/flask")


def test_episode_drops_over_96k():
    huge = "x" * (MAX_CONTEXT_TOKENS * 4 + 100)
    row = build_episode(
        episode_id="big",
        repo="a/b",
        commit="abc",
        teacher="Laguna XS 2.1",
        commission="read it",
        loop_result={"report": huge, "thinking": [], "tool_rounds": [], "messages": [], "raw_steps": []},
    )
    assert row is None


def test_episode_keeps_thinking_out_of_report():
    row = build_episode(
        episode_id="small",
        repo="a/b",
        commit="abc",
        teacher="scripted",
        commission="q",
        loop_result={
            "report": "the function returns 1",
            "thinking": [{"round": 0, "text": "secret chain"}],
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "read_file",
                    "arguments": {"path": "a.py", "start_line": 1, "end_line": 2},
                    "result": {"content": "def answer():\n", "start_line": 1},
                }
            ],
            "messages": [],
            "raw_steps": [],
        },
    )
    assert row is not None
    blob = json.dumps(row)
    assert "secret chain" in blob
    assert row["report"] == "the function returns 1"
    assert row["token_estimate"] == estimate_tokens(blob) or row["token_estimate"] > 0
    assert row["split"] in {"train", "holdout"}


def test_score_checks_existing_page(tmp_path: Path):
    (tmp_path / "a.py").write_text("def answer():\n    return 7\n", encoding="utf-8")
    episode = {
        "report": "a.py defines answer.",
        "token_estimate": 10,
        "tool_rounds": [
            {
                "round": 0,
                "name": "read_file",
                "arguments": {"path": "a.py", "start_line": 1, "end_line": 2},
                "result": {"path": "a.py", "content": "def answer():\n    return 7\n", "start_line": 1, "end_line": 2},
            }
        ],
    }
    scored = score_episode(episode, tmp_path)
    assert scored["valid_tool_names"] == 1.0
    assert scored["stopped_by_round_8"] == 1.0
    assert scored["cited_lines_exist"] == 1.0


def test_parallel_calls_count_toward_the_cap(tmp_path: Path):
    rounds = [
        {"round": index // 3, "name": "grep", "arguments": {"pattern": f"p{index}"}, "result": {"matches": []}}
        for index in range(9)
    ]
    scored = score_episode({"report": "done", "tool_rounds": rounds}, tmp_path)
    assert scored["tool_rounds"] == 9
    assert scored["stopped_by_round_8"] == 0.0


def test_file_grep_and_its_retry_do_not_count(tmp_path: Path):
    target = tmp_path / "addon.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    rejected = {
        "name": "grep",
        "arguments": {"pattern": "VALUE", "path": str(target)},
        "result": {"error": "not_a_directory", "path": str(target)},
    }
    retry = {
        "name": "grep",
        "arguments": {"pattern": "VALUE", "path": str(tmp_path)},
        "result": {"matches": [{"path": "addon.py", "line": 1, "text": "VALUE = 1"}]},
    }
    real = [
        {"name": "read_file", "arguments": {"path": "addon.py", "start_line": "1", "end_line": "1"}, "result": {"content": "VALUE = 1\n", "start_line": 1, "path": "addon.py"}}
        for _ in range(7)
    ]
    rounds = [{"round": 0, **rejected}, {"round": 1, **retry}, *({"round": 2, **item} for item in real)]
    scored = score_episode({"report": "addon.py defines VALUE.", "tool_rounds": rounds}, tmp_path)
    assert scored["stopped_by_round_8"] == 1.0
    assert scored["valid_tool_args"] == 1.0
    assert scored["report_paths_exist"] == 1.0


def test_url_text_is_not_a_report_path(tmp_path: Path):
    (tmp_path / "addon.py").write_text("x = 1\n", encoding="utf-8")
    report = "See addon.py and https://api.polyhaven.com for textures."
    scored = score_episode(
        {
            "report": report,
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "read_file",
                    "arguments": {"path": "addon.py"},
                    "result": {"path": "addon.py", "content": "x = 1\n", "start_line": 1},
                }
            ],
        },
        tmp_path,
    )
    assert scored["report_paths_exist"] == 1.0


def test_score_trace_reads_saved_episodes(tmp_path: Path):
    from playground.run import score_trace
    from playground.traces import append_jsonl

    (tmp_path / "addon.py").write_text("x = 1\n", encoding="utf-8")
    dest = tmp_path / "teacher.jsonl"
    append_jsonl(
        dest,
        {
            "id": "saved",
            "repo": "acme/addon",
            "report": "addon.py exists.",
            "tool_rounds": [
                {
                    "round": 0,
                    "name": "read_file",
                    "arguments": {"path": "addon.py", "start_line": 1, "end_line": 1},
                    "result": {"path": "addon.py", "content": "x = 1\n", "start_line": 1},
                }
            ],
        },
    )
    scored = score_trace(dest, {"saved": tmp_path})
    assert len(scored) == 1
    assert scored[0]["stopped_by_round_8"] == 1.0
