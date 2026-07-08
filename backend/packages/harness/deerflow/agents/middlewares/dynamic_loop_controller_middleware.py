"""Run-level dynamic loop controller middleware.

This middleware complements the existing per-pattern loop detector and
per-tool progress guard. It observes completed tool results before the next
model call, queues short strategy guidance when a run stops making progress,
and can force wrap-up when a stalled run keeps asking for more tools.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import OrderedDict, defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.config.dynamic_loop_config import DynamicLoopConfig

_CHANGE_STRATEGY_GUIDANCE = (
    "[DYNAMIC LOOP] Recent tool calls are not adding useful new information. Change strategy now: use a different tool, inspect a different source, narrow the query, "
    "or explain why enough evidence has been collected. Do not repeat the same action."
)
_WRAP_UP_GUIDANCE = "[DYNAMIC LOOP] You are approaching the useful limit for this run. Stop broad exploration and produce a final answer using the evidence already collected. Mention any remaining uncertainty explicitly."
_NON_INTERACTIVE_GUIDANCE = "[DYNAMIC LOOP] This run is non-interactive. Do not ask the user for clarification. Make the best reasonable assumption, state it, and complete with the available information."
_FORCED_WRAP_UP_MSG = "[DYNAMIC LOOP FORCED WRAP-UP] Repeated no-progress tool results exceeded the dynamic loop safety limit. Producing final answer with the evidence collected so far."

_MAX_TRACKED_RUNS = 1000
_RECENT_FINGERPRINTS = 20
_GUIDANCE_NAME = "dynamic_loop_controller"
_SALIENT_PATTERN = re.compile(
    r"(?P<url>https?://\S+)|(?P<path>(?:/|\.{1,2}/|[A-Za-z0-9_.-]+/)[A-Za-z0-9_./-]+)|(?P<command>`[^`]+`)",
)


@dataclass
class DynamicLoopRunState:
    """Run-scoped loop bookkeeping."""

    observed_model_turns: int = 0
    tool_calls_since_answer: int = 0
    consecutive_no_progress_steps: int = 0
    guidance_sent: set[str] = field(default_factory=set)
    recent_signatures: deque[str] = field(default_factory=lambda: deque(maxlen=_RECENT_FINGERPRINTS))
    seen_tool_messages: set[str] = field(default_factory=set)


class DynamicLoopControllerMiddleware(AgentMiddleware[AgentState]):
    """Detect run-level stagnation and nudge the lead agent to converge."""

    def __init__(self, config: DynamicLoopConfig) -> None:
        super().__init__()
        self._config = config
        self._lock = threading.Lock()
        self._states: OrderedDict[tuple[str, str], DynamicLoopRunState] = OrderedDict()
        self._pending_guidance: dict[tuple[str, str], list[str]] = defaultdict(list)

    @classmethod
    def from_config(cls, config: DynamicLoopConfig) -> DynamicLoopControllerMiddleware:
        return cls(config=config)

    def reset(self, thread_id: str | None = None) -> None:
        with self._lock:
            if thread_id is None:
                self._states.clear()
                self._pending_guidance.clear()
                return

            for key in [key for key in self._states if key[0] == thread_id]:
                self._states.pop(key, None)
                self._pending_guidance.pop(key, None)

    @staticmethod
    def _get_thread_id(runtime: Runtime) -> str:
        context = getattr(runtime, "context", None)
        if isinstance(context, dict) and context.get("thread_id"):
            return str(context["thread_id"])
        return "default"

    @staticmethod
    def _get_run_id(runtime: Runtime) -> str:
        context = getattr(runtime, "context", None)
        if isinstance(context, dict) and context.get("run_id"):
            return str(context["run_id"])
        return str(id(runtime))

    @classmethod
    def _key(cls, runtime: Runtime) -> tuple[str, str]:
        return cls._get_thread_id(runtime), cls._get_run_id(runtime)

    @staticmethod
    def _is_non_interactive(runtime: Runtime) -> bool:
        context = getattr(runtime, "context", None)
        if not isinstance(context, dict):
            return False
        nested = context.get("context")
        return bool(context.get("non_interactive") or (isinstance(nested, dict) and nested.get("non_interactive")))

    @staticmethod
    def _get_recursion_limit(runtime: Runtime) -> int | None:
        config = getattr(runtime, "config", None)
        if not isinstance(config, dict):
            return None

        value = config.get("recursion_limit")
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    def _state_for_key_locked(self, key: tuple[str, str]) -> DynamicLoopRunState:
        state = self._states.get(key)
        if state is None:
            state = DynamicLoopRunState()
            self._states[key] = state
        self._states.move_to_end(key)
        while len(self._states) > _MAX_TRACKED_RUNS:
            evicted_key, _ = self._states.popitem(last=False)
            self._pending_guidance.pop(evicted_key, None)
        return state

    @staticmethod
    def _text_content(message: ToolMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    value = item.get("text") or item.get("content")
                    if value is not None:
                        parts.append(str(value))
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _message_key(message: ToolMessage) -> str:
        if message.id:
            return f"id:{message.id}"
        text = DynamicLoopControllerMiddleware._text_content(message)
        digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"tool:{message.tool_call_id}:{digest}"

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalized = " ".join(text.strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def _has_salient_reference(text: str) -> bool:
        return bool(_SALIENT_PATTERN.search(text))

    def _counts_as_progress(self, message: ToolMessage, run_state: DynamicLoopRunState) -> bool:
        text = self._text_content(message)
        if not text.strip():
            return False

        fingerprint = self._fingerprint(text)
        is_new_fingerprint = fingerprint not in run_state.recent_signatures
        has_enough_text = len(text.strip()) >= self._config.min_new_info_chars
        has_salient_reference = self._has_salient_reference(text)

        metadata = getattr(message, "additional_kwargs", {}) or {}
        tool_meta = metadata.get("deerflow_tool_meta") or {}
        status = tool_meta.get("status")
        is_recoverable_error = bool(tool_meta.get("recoverable_by_model")) and status == "error"

        return is_new_fingerprint and not is_recoverable_error and (has_enough_text or has_salient_reference or status in {"success", "partial_success"})

    def _queue_guidance_locked(self, key: tuple[str, str], run_state: DynamicLoopRunState, guidance: str) -> None:
        if guidance in run_state.guidance_sent:
            return
        pending = self._pending_guidance[key]
        if guidance not in pending:
            pending.append(guidance)
        run_state.guidance_sent.add(guidance)

    def _observe_tool_message_locked(self, message: ToolMessage, runtime: Runtime, key: tuple[str, str], run_state: DynamicLoopRunState) -> None:
        message_key = self._message_key(message)
        if message_key in run_state.seen_tool_messages:
            return
        run_state.seen_tool_messages.add(message_key)

        text = self._text_content(message)
        fingerprint = self._fingerprint(text)
        progressed = self._counts_as_progress(message, run_state)
        run_state.recent_signatures.append(fingerprint)

        if progressed:
            run_state.consecutive_no_progress_steps = 0
            return

        run_state.consecutive_no_progress_steps += 1
        if run_state.consecutive_no_progress_steps >= self._config.stagnation_warn_steps:
            if self._is_non_interactive(runtime) and self._config.force_wrap_up_on_non_interactive_stagnation:
                self._queue_guidance_locked(key, run_state, _NON_INTERACTIVE_GUIDANCE)
            else:
                self._queue_guidance_locked(key, run_state, _CHANGE_STRATEGY_GUIDANCE)

    def _observe_request_messages(self, messages: list[Any], runtime: Runtime) -> None:
        if not self._config.enabled:
            return

        key = self._key(runtime)
        with self._lock:
            run_state = self._state_for_key_locked(key)
            for message in messages:
                if isinstance(message, ToolMessage):
                    self._observe_tool_message_locked(message, runtime, key, run_state)

    @staticmethod
    def _append_text(content: str | list[dict | str] | None, stop_msg: str) -> str | list[dict | str]:
        if content is None:
            return stop_msg
        if isinstance(content, str):
            if content:
                return f"{content}\n\n{stop_msg}"
            return stop_msg
        if isinstance(content, list):
            new_content = list(content)
            new_content.append({"type": "text", "text": f"\n\n{stop_msg}"})
            return new_content
        return f"{content}\n\n{stop_msg}"

    def _build_hard_stop_update(self, message: AIMessage) -> dict[str, Any]:
        kwargs = dict(message.additional_kwargs) if message.additional_kwargs else {}
        kwargs.pop("tool_calls", None)
        kwargs.pop("function_call", None)

        response_metadata = dict(getattr(message, "response_metadata", {}) or {})
        if response_metadata.get("finish_reason") == "tool_calls":
            response_metadata["finish_reason"] = "stop"

        stopped = message.model_copy(
            update={
                "content": self._append_text(message.content, _FORCED_WRAP_UP_MSG),
                "tool_calls": [],
                "additional_kwargs": kwargs,
                "response_metadata": response_metadata,
            }
        )
        return {"messages": [stopped]}

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        if not self._config.enabled:
            return None

        messages = state.get("messages", [])
        if not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None

        key = self._key(runtime)
        tool_calls = getattr(last_message, "tool_calls", None) or []
        with self._lock:
            run_state = self._state_for_key_locked(key)
            run_state.observed_model_turns += 1
            if not tool_calls:
                run_state.tool_calls_since_answer = 0
                return None

            if run_state.consecutive_no_progress_steps >= self._config.stagnation_hard_limit:
                return self._build_hard_stop_update(last_message)

            run_state.tool_calls_since_answer += len(tool_calls)
            if run_state.tool_calls_since_answer >= self._config.max_tool_calls_without_answer:
                self._queue_guidance_locked(key, run_state, _WRAP_UP_GUIDANCE)

            recursion_limit = self._get_recursion_limit(runtime)
            if recursion_limit is not None:
                remaining_turns = recursion_limit - run_state.observed_model_turns
                if remaining_turns <= self._config.wrap_up_when_recursion_remaining:
                    self._queue_guidance_locked(key, run_state, _WRAP_UP_GUIDANCE)

        return None

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self.after_model(state, runtime)

    def _drain_guidance(self, runtime: Runtime) -> list[str]:
        if not self._config.enabled:
            return []
        key = self._key(runtime)
        with self._lock:
            guidance = self._pending_guidance.pop(key, None)
        return guidance or []

    @staticmethod
    def _inject_guidance(request: ModelRequest, guidance: list[str]) -> ModelRequest:
        if not guidance:
            return request
        message = HumanMessage(content="\n\n".join(guidance), name=_GUIDANCE_NAME)
        return request.override(messages=list(getattr(request, "messages", [])) + [message])

    @override
    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        self._observe_request_messages(list(getattr(request, "messages", [])), request.runtime)
        guidance = self._drain_guidance(request.runtime)
        return handler(self._inject_guidance(request, guidance))

    @override
    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelCallResult:
        self._observe_request_messages(list(getattr(request, "messages", [])), request.runtime)
        guidance = self._drain_guidance(request.runtime)
        return await handler(self._inject_guidance(request, guidance))

    @override
    def after_agent(self, state: AgentState, runtime: Runtime) -> None:
        if not self._config.enabled:
            return
        key = self._key(runtime)
        with self._lock:
            self._states.pop(key, None)
            self._pending_guidance.pop(key, None)

    @override
    async def aafter_agent(self, state: AgentState, runtime: Runtime) -> None:
        self.after_agent(state, runtime)
