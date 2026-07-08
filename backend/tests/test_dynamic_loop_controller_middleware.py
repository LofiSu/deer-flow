"""Tests for DynamicLoopControllerMiddleware."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.dynamic_loop_controller_middleware import (
    _CHANGE_STRATEGY_GUIDANCE,
    _FORCED_WRAP_UP_MSG,
    _NON_INTERACTIVE_GUIDANCE,
    _WRAP_UP_GUIDANCE,
    DynamicLoopControllerMiddleware,
)
from deerflow.config.dynamic_loop_config import DynamicLoopConfig


def _make_runtime(thread_id: str = "thread-1", run_id: str = "run-1", config: dict | None = None, **context):
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id, "run_id": run_id, **context}
    runtime.config = config or {}
    return runtime


def _make_request(messages, runtime):
    request = MagicMock()
    request.messages = list(messages)
    request.runtime = runtime
    request.override = lambda **updates: _override_request(request, updates)
    return request


def _override_request(request, updates):
    new = MagicMock()
    new.messages = updates.get("messages", request.messages)
    new.runtime = updates.get("runtime", request.runtime)
    new.override = lambda **u: _override_request(new, u)
    return new


def _capture_handler():
    captured: list = []

    def handler(req):
        captured.append(req)
        return MagicMock()

    return captured, handler


def _tool_message(content: str, tool_call_id: str = "call-1", name: str = "bash") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tool_call_id, name=name)


def test_substantial_new_tool_output_resets_no_progress_count() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(stagnation_warn_steps=2, min_new_info_chars=12))
    runtime = _make_runtime()

    mw._observe_request_messages([_tool_message("")], runtime)
    assert mw._states[("thread-1", "run-1")].consecutive_no_progress_steps == 1

    mw._observe_request_messages([_tool_message("new evidence with enough detail", tool_call_id="call-2")], runtime)

    assert mw._states[("thread-1", "run-1")].consecutive_no_progress_steps == 0


def test_repeated_no_progress_queues_change_strategy_guidance_once() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(stagnation_warn_steps=2, stagnation_hard_limit=5))
    runtime = _make_runtime()

    mw._observe_request_messages([_tool_message("")], runtime)
    mw._observe_request_messages([_tool_message("", tool_call_id="call-2")], runtime)
    mw._observe_request_messages([_tool_message("", tool_call_id="call-3")], runtime)

    assert mw._pending_guidance[("thread-1", "run-1")] == [_CHANGE_STRATEGY_GUIDANCE]


def test_queued_guidance_is_injected_from_wrap_model_call() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(stagnation_warn_steps=1))
    runtime = _make_runtime()
    mw._observe_request_messages([_tool_message("")], runtime)
    request = _make_request([HumanMessage(content="start"), _tool_message("")], runtime)
    captured, handler = _capture_handler()

    mw.wrap_model_call(request, handler)

    sent_messages = captured[0].messages
    assert isinstance(sent_messages[-1], HumanMessage)
    assert sent_messages[-1].content == _CHANGE_STRATEGY_GUIDANCE
    assert mw._pending_guidance.get(("thread-1", "run-1")) is None


def test_non_interactive_stagnation_queues_non_interactive_wrap_up() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(stagnation_warn_steps=1))
    runtime = _make_runtime(non_interactive=True)

    mw._observe_request_messages([_tool_message("")], runtime)

    assert mw._pending_guidance[("thread-1", "run-1")] == [_NON_INTERACTIVE_GUIDANCE]


def test_tool_call_budget_queues_wrap_up_guidance() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(max_tool_calls_without_answer=2))
    runtime = _make_runtime()
    ai_message = AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "pwd"}, "id": "call-1"}])

    assert mw.after_model({"messages": [ai_message]}, runtime) is None
    assert mw.after_model({"messages": [ai_message]}, runtime) is None

    assert mw._pending_guidance[("thread-1", "run-1")] == [_WRAP_UP_GUIDANCE]


def test_recursion_pressure_queues_wrap_up_guidance_when_limit_is_visible() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(max_tool_calls_without_answer=100, wrap_up_when_recursion_remaining=3))
    runtime = _make_runtime(config={"recursion_limit": 10})
    ai_message = AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "pwd"}, "id": "call-1"}])

    for _ in range(7):
        assert mw.after_model({"messages": [ai_message]}, runtime) is None

    assert mw._pending_guidance[("thread-1", "run-1")] == [_WRAP_UP_GUIDANCE]


def test_recursion_pressure_is_skipped_when_limit_is_not_visible() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(max_tool_calls_without_answer=100, wrap_up_when_recursion_remaining=3))
    runtime = _make_runtime()
    ai_message = AIMessage(content="", tool_calls=[{"name": "bash", "args": {"command": "pwd"}, "id": "call-1"}])

    for _ in range(20):
        assert mw.after_model({"messages": [ai_message]}, runtime) is None

    assert mw._pending_guidance.get(("thread-1", "run-1")) is None


def test_hard_limit_strips_tool_calls_and_provider_tool_metadata() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(stagnation_warn_steps=1, stagnation_hard_limit=2))
    runtime = _make_runtime()
    mw._observe_request_messages([_tool_message("")], runtime)
    mw._observe_request_messages([_tool_message("", tool_call_id="call-2")], runtime)
    ai_message = AIMessage(
        content="still trying",
        tool_calls=[{"name": "bash", "args": {"command": "pwd"}, "id": "call-1"}],
        additional_kwargs={"tool_calls": [{"id": "raw"}], "function_call": {"name": "legacy"}},
        response_metadata={"finish_reason": "tool_calls"},
    )

    update = mw.after_model({"messages": [ai_message]}, runtime)

    stopped = update["messages"][0]
    assert stopped.tool_calls == []
    assert "tool_calls" not in stopped.additional_kwargs
    assert "function_call" not in stopped.additional_kwargs
    assert stopped.response_metadata["finish_reason"] == "stop"
    assert _FORCED_WRAP_UP_MSG in stopped.content


def test_after_agent_clears_run_state() -> None:
    mw = DynamicLoopControllerMiddleware.from_config(DynamicLoopConfig(stagnation_warn_steps=1))
    runtime = _make_runtime()
    mw._observe_request_messages([_tool_message("")], runtime)

    mw.after_agent({"messages": []}, runtime)

    assert ("thread-1", "run-1") not in mw._states
    assert ("thread-1", "run-1") not in mw._pending_guidance
