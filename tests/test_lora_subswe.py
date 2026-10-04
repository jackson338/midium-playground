import json
from pathlib import Path

from playground.lora_reader import assistant_message_from_text, transcript_from_loop_messages
from playground.subswe import fetch_subswe_repos, score_model


def test_json_tool_turn_becomes_a_tool_call():
    message = assistant_message_from_text(
        '{"name": "grep", "arguments": {"pattern": "Logger"}}'
    )
    assert message["tool_calls"][0]["function"]["name"] == "grep"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["pattern"] == "Logger"
    assert message["content"] == ""


def test_plain_text_is_the_report():
    message = assistant_message_from_text("Logger is defined in loguru/_logger.py.")
    assert "tool_calls" not in message
    assert "loguru/_logger.py" in message["content"]


def test_thinking_text_is_not_a_tool_call():
    message = assistant_message_from_text(
        "<think>{\"name\": \"grep\", \"arguments\": {\"pattern\": \"x\"}}</think>\n"
        "The class is in loguru/_logger.py."
    )
    assert "tool_calls" not in message
    assert message["reasoning"].startswith('{"name"')
    assert "loguru/_logger.py" in message["content"]


def test_fetch_repos_checks_out_the_pinned_commit(tmp_path: Path):
    calls: list[list[str]] = []

    def runner(cmd, check=False):
        del check
        calls.append(list(cmd))

        class Proc:
            returncode = 0

        return Proc()

    fetch_subswe_repos(
        [
            {
                "repo": "Delgan/loguru",
                "commit": "0412a60d6aff63a4ed009c7445993b42d294c2f1",
                "path": "data/repos/Delgan__loguru",
            }
        ],
        dest_root=tmp_path,
        runner=runner,
    )
    joined = [" ".join(cmd) for cmd in calls]
    assert any("https://github.com/Delgan/loguru.git" in cmd for cmd in joined)
    assert any("fetch --depth 1 origin 0412a60d6aff63a4ed009c7445993b42d294c2f1" in cmd for cmd in joined)
    assert any("checkout --detach 0412a60d6aff63a4ed009c7445993b42d294c2f1" in cmd for cmd in joined)


def test_score_reads_only_the_named_model(tmp_path: Path, monkeypatch):
    from playground import subswe

    monkeypatch.setattr(subswe, "SUBSWE_DIR", tmp_path)
    monkeypatch.setattr(
        subswe,
        "load_subswe_tasks",
        lambda: [
            {
                "id": "locate-one",
                "kind": "locate",
                "path": str(tmp_path),
                "gold": {"paths": ["a.py"], "symbol": "Widget"},
            }
        ],
    )
    (tmp_path / "a.py").write_text("class Widget:\n", encoding="utf-8")
    (tmp_path / "gemma-4-e4b-lora.run1.jsonl").write_text(
        json.dumps(
            {
                "id": "locate-one",
                "error": None,
                "report": "Widget is in a.py",
                "stop_reason": "report",
                "tool_rounds": [
                    {"name": "grep", "arguments": {"pattern": "Widget"}, "result": {"matches": []}}
                ],
                "semantic": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scored = score_model("Gemma 4 E4B LoRA", 1)
    assert scored["pass_rate"] == 1.0
    assert scored["error_count"] == 0
