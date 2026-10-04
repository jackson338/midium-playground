import json

import pytest

from playground.compare_lora import drop_previous_lora_trace, print_pair, repos_present
from playground.gemma_turns import (
    assistant_from_gemma_text,
    build_checked_tool_train_set,
    build_tool_train_set,
    episode_messages,
    next_turn,
    require_tool_call_loss,
)
from playground.lora_reader import LoraReader
from playground.subswe import F16_MODEL, LORA_MODEL, model_spec, trace_path


def _render(messages: list[dict]) -> str:
    chunks = []
    for message in messages:
        chunks.append(message.get("content") or "")
        for call in message.get("tool_calls") or []:
            function = call["function"]
            chunks.append(f"<|tool_call>call:{function['name']}<tool_call|>")
    return "\n".join(chunks)


def _row(split: str, page: str, commission: str = "Where is Logger?") -> dict:
    return {
        "id": split + commission[:8],
        "split": split,
        "repo": "Delgan/loguru",
        "commission": commission,
        "report": "Logger is in loguru/_logger.py.",
        "thinking": [{"round": 0, "text": "secret thought"}],
        "tool_rounds": [
            {
                "name": "grep",
                "arguments": {"pattern": "Logger"},
                "result": {"matches": []},
            },
            {
                "name": "read_file",
                "arguments": {"path": "a.py", "start_line": 1, "end_line": 2},
                "result": {"content": page},
            },
        ],
    }


def test_tool_messages_are_real_tool_calls():
    messages = episode_messages(_row("train", "PAGE"), ["PAGE"], "SKILL")
    assert messages[0] == {"role": "system", "content": "SKILL"}
    call = messages[2]["tool_calls"][0]["function"]
    assert call["name"] == "grep"
    assert call["arguments"] == {"pattern": "Logger"}
    assert messages[2]["content"] == ""
    assert messages[-1]["role"] == "assistant"
    assert "tool_calls" not in messages[-1]
    assert "secret thought" not in json.dumps(messages)


def test_train_set_keeps_the_first_hundred_that_fit():
    rows = [_row("holdout", "H", commission="holdout question")]
    rows.extend(_row("train", "p", commission=f"question {index}") for index in range(120))
    packed, skipped = build_tool_train_set(rows, 100000, lambda text: len(text), _render, n=100)
    assert skipped == 1
    assert len(packed) == 100
    assert "holdout question" not in packed[0]["text"]
    assert "<|tool_call>call:grep" in packed[0]["text"]
    assert '{"name"' not in packed[0]["text"]


def test_read_pages_are_dropped_to_fit():
    packed, skipped = build_tool_train_set(
        [_row("train", "x" * 8000)],
        4000,
        lambda text: len(text),
        _render,
        n=100,
    )
    assert skipped == 0
    assert "x" * 100 not in packed[0]["text"]
    assert "<|tool_call>call:read_file" in packed[0]["text"]


def test_loss_check_stops_before_the_hundred_when_the_template_is_plain_json():
    seen: list[str] = []

    def render(messages: list[dict]) -> str:
        seen.append(messages[1]["content"])
        return '{"name": "grep", "arguments": {"pattern": "Logger"}}'

    rows = [_row("train", "p", commission=f"question {index}") for index in range(10)]
    with pytest.raises(SystemExit, match="plain-text adapter"):
        build_checked_tool_train_set(rows, 100000, lambda text: len(text), render, n=100)
    assert seen
    assert set(seen) == {"question 0"}


def test_loss_check_accepts_a_gemma_tool_call_string():
    require_tool_call_loss('<|tool_call>call:grep{pattern:<|"|>Logger<|"|>}<tool_call|>')
    with pytest.raises(SystemExit):
        require_tool_call_loss("assistant\n" + '{"name": "grep"}')
    with pytest.raises(SystemExit):
        require_tool_call_loss("no tool call here")


def test_loss_check_allows_name_inside_a_tool_response():
    require_tool_call_loss(
        '<|tool_call>call:list_files{path:<|"|>src<|"|>}<tool_call|>'
        '<|tool_response>response:list_files{value:<|"|>'
        '[{"name": "inference", "path": "inference"}]'
        '<|"|>}<tool_response|>'
    )
    with pytest.raises(SystemExit, match="plain-text adapter"):
        require_tool_call_loss(
            '<|tool_call>call:grep{}<tool_call|>'
            '{"name": "grep", "arguments": {"pattern": "Logger"}}'
        )


def test_checked_train_set_keeps_tool_call_examples():
    rows = [_row("train", "p", commission=f"question {index}") for index in range(3)]
    packed, _skipped = build_checked_tool_train_set(rows, 100000, lambda text: len(text), _render, n=3)
    assert len(packed) == 3
    assert "<|tool_call>call:grep" in packed[0]["text"]
    assert '{"name"' not in packed[0]["text"]


def test_old_adapter_directory_is_refused():
    from playground.train import CHECKPOINT_DIR, TOOLS_CHECKPOINT_DIR, resolve_tools_checkpoint

    assert resolve_tools_checkpoint(None) == TOOLS_CHECKPOINT_DIR
    with pytest.raises(SystemExit, match="e4b-32k-lora"):
        resolve_tools_checkpoint(CHECKPOINT_DIR)


def test_overlong_skeleton_is_skipped():
    huge = _row("train", "PAGE", commission="Q" * 20000)
    with pytest.raises(SystemExit):
        build_tool_train_set([huge], 1000, lambda text: len(text), _render)


def test_first_gemma_call_ignores_the_fake_tool_line():
    text = (
        '<|tool_call>call:grep{pattern:<|"|>Logger<|"|>}<tool_call|>\n'
        'tool\n{"content":"FAKE_RESULT"}\n'
        '<|tool_call>call:read_file{path:<|"|>a.py<|"|>}<tool_call|>'
    )
    message = assistant_from_gemma_text(text)
    assert message["tool_calls"][0]["function"]["name"] == "grep"
    arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"pattern": "Logger"}
    blob = json.dumps(message)
    assert "FAKE_RESULT" not in blob
    assert "read_file" not in blob
    assert message["content"] == ""


def test_integer_arguments_stay_integers():
    text = '<|tool_call>call:read_file{path:<|"|>a.py<|"|>,start_line:12,end_line:40}<tool_call|>'
    arguments = json.loads(
        assistant_from_gemma_text(text)["tool_calls"][0]["function"]["arguments"]
    )
    assert arguments == {"path": "a.py", "start_line": 12, "end_line": 40}


def test_plain_text_is_the_report():
    message = assistant_from_gemma_text("Logger is defined in loguru/_logger.py.")
    assert "tool_calls" not in message
    assert "loguru/_logger.py" in message["content"]


def test_unknown_tool_is_recorded_once_then_the_loop_stops():
    message, stopped = next_turn(
        '<|tool_call>call:not_a_tool{}<tool_call|>\ntool\n{"content":"FAKE_RESULT"}',
        False,
    )
    assert message["tool_calls"][0]["function"]["name"] == "not_a_tool"
    assert "FAKE_RESULT" not in json.dumps(message)
    assert stopped
    follow, still = next_turn('<|tool_call>call:grep{pattern:<|"|>x<|"|>}<tool_call|>', True)
    assert "tool_calls" not in follow
    assert follow["content"] == "Stopped."
    assert still


def test_reader_returns_only_the_first_call(monkeypatch):
    reader = LoraReader(None)
    reader._model = object()
    reader._processor = object()
    monkeypatch.setattr("playground.lora_reader.render_prompt", lambda *args, **kwargs: "PROMPT")
    generation = (
        '<|tool_call>call:grep{pattern:<|"|>Logger<|"|>}<tool_call|>\n'
        'tool\n{"content":"FAKE_RESULT"}'
    )
    monkeypatch.setattr(
        "playground.lora_reader.generate_continuation",
        lambda *args, **kwargs: generation,
    )
    message = reader._complete_sync([{"role": "user", "content": "find Logger"}], None)
    assert message["tool_calls"][0]["function"]["name"] == "grep"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"])["pattern"] == "Logger"
    assert "FAKE_RESULT" not in json.dumps(message)
    assert reader._stopped is False


def test_trace_names_and_routes():
    assert trace_path(F16_MODEL, 1).name == "gemma-4-e4b-f16.run1.jsonl"
    assert trace_path(LORA_MODEL, 1).name == "gemma-4-e4b-lora.run1.jsonl"
    assert trace_path("Gemma 4 E4B", 1).name == "gemma-4-e4b.run1.jsonl"
    assert trace_path(F16_MODEL, 1).name != trace_path("Gemma 4 E4B", 1).name
    assert model_spec(F16_MODEL) == ("lora", 1)
    assert model_spec(LORA_MODEL) == ("lora", 1)


def test_drop_previous_lora_trace(tmp_path):
    old = tmp_path / "gemma-4-e4b-lora.run1.jsonl"
    old.write_text("old\n", encoding="utf-8")
    drop_previous_lora_trace(old)
    assert not old.exists()
    drop_previous_lora_trace(old)


def test_repos_present(tmp_path):
    assert repos_present([{"path": str(tmp_path / "missing")}]) is False
    repo = tmp_path / "repo"
    repo.mkdir()
    assert repos_present([{"path": str(repo)}]) is True


def test_print_pair_lists_both_rates_without_an_average(capsys):
    def row(model: str, rate: float) -> dict:
        return {
            "model": model,
            "pass_rate": rate,
            "passed": 1,
            "attempted": 4,
            "call_count": 3,
            "by_kind": {"locate": {"passed": 1, "attempted": 4}},
            "tasks": [
                {
                    "id": "locate-one",
                    "error": None,
                    "deterministic": {"passed": False, "valid_tools": False, "stopped_clean": True},
                }
            ],
        }

    print_pair(row(F16_MODEL, 0.25), row(LORA_MODEL, 0.5))
    out = capsys.readouterr().out
    assert f"{F16_MODEL}: pass_rate=0.25" in out
    assert f"{LORA_MODEL}: pass_rate=0.5" in out
    assert "locate 1/4" in out
    assert "fail locate-one: valid_tools" in out
    assert "average" not in out.lower()
