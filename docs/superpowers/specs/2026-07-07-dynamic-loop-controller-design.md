# RFC: DeerFlow Dynamic Loop Controller Design

**Date**: 2026-07-07
**Status**: Accepted for P0 implementation
**Scope**: Middleware-based dynamic loop control for the existing lead-agent ReAct loop

---

## Problem Statement

DeerFlow already runs a dynamic model/tool loop through `create_agent(...)`: the model emits tool calls, tools execute, tool results return to the model, and the loop continues until the model produces a final answer or LangGraph reaches a stop condition.

The current safety layer is effective but narrow:

1. `recursion_limit` prevents unbounded graph super-steps.
2. `LoopDetectionMiddleware` stops repeated tool-call patterns and excessive calls to one tool type.
3. `ToolProgressMiddleware` evaluates result quality for individual tools.
4. `TokenBudgetMiddleware` can enforce token ceilings.

What is missing is run-level loop control. The agent can still spend many turns gathering low-value information, drift across tools without enough new evidence, or continue exploring when the useful next action is to synthesize. The first implementation should make the existing loop more adaptive without creating a second agent runtime or replacing LangGraph's built-in ReAct loop.

## RFC Decision

Implement a conservative lead-agent `DynamicLoopControllerMiddleware` as the P0 solution. Do not replace the current `create_agent(...)` ReAct loop with an explicit `StateGraph` in this phase.

The target behavior is:

```text
When a long-running lead-agent run keeps using tools but recent tool results no longer add new information, DeerFlow should guide the model to change strategy or wrap up, and should safely force finalization after configured no-progress limits without breaking provider tool-call message validity.
```

This RFC deliberately treats explicit `StateGraph` dynamic-loop orchestration as a future option that must be justified by benchmark evidence after the middleware and progress-ledger layers are measured.

## Goals

1. Keep the existing `create_agent(...)` lead-agent loop as the execution engine.
2. Add deterministic run-level guidance that detects stagnation, progress, and budget pressure.
3. Improve long-task quality by nudging the model to change strategy, verify, or wrap up at the right time.
4. Preserve current hard safety controls: recursion clamping, loop detection, token budget, clarification handling, and tool-call pairing semantics.
5. Make behavior configurable and testable without adding evaluator LLM calls.

## Definition of Done

P0 is complete only when all of these are true:

1. RFC and implementation plan describe the same architecture, file boundaries, configuration, risks, and verification commands.
2. Every public config field in `dynamic_loop` either has implemented behavior or is explicitly marked future-only. P0 has no inert config fields.
3. The lead-agent middleware chain includes `DynamicLoopControllerMiddleware` by default and omits it when `dynamic_loop.enabled=false`.
4. The controller observes completed `ToolMessage`s at model-call boundaries and does not insert messages between `AIMessage(tool_calls=...)` and matching `ToolMessage`s.
5. The controller emits change-strategy guidance after warning-level stagnation, non-interactive wrap-up guidance when stalled with `non_interactive=true`, and wrap-up guidance under configured budget pressure.
6. Hard-stop behavior clears structured `tool_calls`, raw provider tool-call metadata, legacy `function_call`, and `finish_reason="tool_calls"` before forcing finalization.
7. Run-scoped controller state is bounded and cleared at run end.
8. Focused tests and lint/format checks pass, with coverage for config validation, middleware behavior, lead-agent chain placement, and existing loop/token safety middleware compatibility.

## Non-Goals

This design intentionally does not include:

1. A new explicit `StateGraph` such as `plan -> execute -> evaluate -> revise -> final`.
2. Periodic evaluator-model calls to judge progress.
3. Frontend UI changes.
4. Changes to Gateway run lifecycle, `RunManager`, `StreamBridge`, or persistence schemas.
5. Replacement of `LoopDetectionMiddleware` or `ToolProgressMiddleware`.
6. Semantic proof that the final answer is correct. The controller only improves loop strategy and termination behavior.

P0 also does not claim to close all long-horizon quality issues. Research-search, SWE-like coding, scientific-computing, and scheduled-task trajectory benchmarks are required before claiming product-level success.

## Chosen Architecture

Add a new `DynamicLoopControllerMiddleware` to the lead-agent middleware chain.

The controller observes model/tool loop behavior and injects short, hidden `HumanMessage` guidance at the next model call. It can also force a final answer in severe no-progress cases by stripping tool calls from the latest `AIMessage`, using the same safety pattern as `LoopDetectionMiddleware`.

The middleware is a run-level coordinator:

- `ToolProgressMiddleware` remains the per-tool result-quality guard.
- `LoopDetectionMiddleware` remains the repeated-call hard safety guard.
- `DynamicLoopControllerMiddleware` decides whether the run as a whole is still making useful progress.

## Placement

Register the new middleware in `backend/packages/harness/deerflow/agents/lead_agent/agent.py` inside `build_middlewares`.

Recommended order:

```text
SystemMessageCoalescingMiddleware
SubagentLimitMiddleware
DynamicLoopControllerMiddleware
LoopDetectionMiddleware
TokenBudgetMiddleware
custom middlewares
SafetyFinishReasonMiddleware
ClarificationMiddleware
```

This placement lets the controller provide strategy guidance before the existing loop detector reaches hard-stop thresholds. `LoopDetectionMiddleware` still remains the final repeated-pattern safety valve.

The first implementation applies to the lead agent only. Subagent support is explicitly out of scope for this design and should require a separate follow-up spec if the lead-agent behavior proves stable.

## Configuration

Add `backend/packages/harness/deerflow/config/dynamic_loop_config.py`:

```python
from pydantic import BaseModel, Field, model_validator


class DynamicLoopConfig(BaseModel):
    enabled: bool = Field(default=True)
    stagnation_warn_steps: int = Field(default=3, ge=1)
    stagnation_hard_limit: int = Field(default=6, ge=1)
    max_tool_calls_without_answer: int = Field(default=40, ge=1)
    wrap_up_when_recursion_remaining: int = Field(default=8, ge=1)
    min_new_info_chars: int = Field(default=120, ge=1)
    force_wrap_up_on_non_interactive_stagnation: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "DynamicLoopConfig":
        if self.stagnation_hard_limit < self.stagnation_warn_steps:
            raise ValueError("stagnation_hard_limit must be >= stagnation_warn_steps")
        return self
```

Add it to `AppConfig`:

```python
dynamic_loop: DynamicLoopConfig = Field(
    default_factory=DynamicLoopConfig,
    description="Run-level dynamic loop control for the lead agent",
)
```

Update `config.example.yaml` with a documented `dynamic_loop` block and bump `config_version`, because this is a config schema change.

## Runtime State

The controller should keep run-scoped bookkeeping in memory, keyed by `(thread_id, run_id)`. It should not add persisted `ThreadState` fields in the first version.

Tracked fields:

```python
@dataclass
class DynamicLoopRunState:
    observed_model_turns: int = 0
    tool_calls_since_answer: int = 0
    consecutive_no_progress_steps: int = 0
    guidance_sent: set[str] = field(default_factory=set)
    recent_signatures: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    seen_tool_messages: set[str] = field(default_factory=set)
```

Rationale:

1. The data is operational control state, not durable user-facing state.
2. If the process restarts, existing recursion limits and loop detection still protect the run.
3. Avoiding persisted state keeps checkpoint compatibility simple.

## Progress Heuristics

The first version uses deterministic heuristics only.

### Progress Signals

A tool result counts as progress when at least one of these is true:

1. It contains at least `min_new_info_chars` of non-error text not recently seen.
2. Its metadata indicates success or partial success and the normalized content fingerprint is new.
3. It references a new path, URL, query, command, artifact, or viewed image not present in recent signatures.
4. It is a terminal subagent result with a non-empty result summary.

### No-Progress Signals

A tool result counts as no progress when it is:

1. Empty or whitespace-only.
2. A recoverable error with no new actionable detail.
3. A repeated error of the same category and tool.
4. A repeated content fingerprint.
5. Below `min_new_info_chars` and not tied to a new salient path, URL, command, or artifact.

### Budget Pressure Signals

The controller should encourage wrap-up when:

1. `tool_calls_since_answer >= max_tool_calls_without_answer`.
2. `consecutive_no_progress_steps >= stagnation_warn_steps`.
3. The run appears close to `recursion_limit`, when remaining budget can be derived from runtime config and observed model/tool turns.
4. `non_interactive=true` and the controller would otherwise suggest asking the user.

The first implementation treats recursion remaining as best-effort. If `runtime.config.recursion_limit` or equivalent runtime metadata is absent, the controller must skip this check rather than guessing. When present, the controller should use observed model turns as a conservative approximation and queue wrap-up guidance when `recursion_limit - observed_model_turns <= wrap_up_when_recursion_remaining`.

## Guidance Behavior

The controller queues guidance in `after_model` or tool wrappers and injects it in `wrap_model_call`, appended after existing messages as a hidden `HumanMessage`.

Guidance messages should be concise and operational:

### Change Strategy

Used after warning-level stagnation:

```text
[DYNAMIC LOOP] Recent tool calls are not adding useful new information. Change strategy now: use a different tool, inspect a different source, narrow the query, or explain why enough evidence has been collected. Do not repeat the same action.
```

### Wrap Up

Used near budget limits or after enough exploration:

```text
[DYNAMIC LOOP] You are approaching the useful limit for this run. Stop broad exploration and produce a final answer using the evidence already collected. Mention any remaining uncertainty explicitly.
```

### Non-Interactive Wrap Up

Used when `context.non_interactive=true` and progress stalls:

```text
[DYNAMIC LOOP] This run is non-interactive. Do not ask the user for clarification. Make the best reasonable assumption, state it, and complete with the available information.
```

Warnings are transient and run-scoped. They must be cleared in `after_agent` so a stale warning does not leak into a later run on the same thread.

## Hard Stop Behavior

When `consecutive_no_progress_steps >= stagnation_hard_limit`, the controller may force a final answer.

Hard stop semantics:

1. Apply only after the last model response includes `tool_calls`.
2. Clear structured `tool_calls`.
3. Remove raw provider tool-call metadata from `additional_kwargs`.
4. If `response_metadata.finish_reason == "tool_calls"`, change it to `"stop"`.
5. Append a short forced-wrap-up note to the assistant content.

This mirrors the existing `LoopDetectionMiddleware` hard-stop pattern and preserves provider tool-call pairing rules.

## Tool-Call Pairing Rule

The controller must not insert a message between an `AIMessage(tool_calls=...)` and the corresponding `ToolMessage` responses.

Therefore:

1. Guidance is queued when detected.
2. Guidance is injected only in `wrap_model_call`, after previous tool results are already in the outgoing request.
3. Hard stop may mutate the current `AIMessage` only by clearing tool calls and raw tool-call metadata, making the message a plain assistant response.

This is the same compatibility constraint documented by `LoopDetectionMiddleware`.

## Implementation Files

New files:

- `backend/packages/harness/deerflow/config/dynamic_loop_config.py`
- `backend/packages/harness/deerflow/agents/middlewares/dynamic_loop_controller_middleware.py`
- `backend/tests/test_dynamic_loop_config.py`
- `backend/tests/test_dynamic_loop_controller_middleware.py`

Modified files:

- `backend/packages/harness/deerflow/config/__init__.py`
- `backend/packages/harness/deerflow/config/app_config.py`
- `backend/packages/harness/deerflow/agents/lead_agent/agent.py`
- `backend/tests/test_lead_agent_model_resolution.py`
- `config.example.yaml`
- `backend/AGENTS.md`

`README.md` only needs an update if the final implementation exposes user-facing configuration in a way end users are expected to tune.

## Testing Plan

Add focused unit tests for config validation:

1. Defaults are enabled and sane.
2. `stagnation_hard_limit < stagnation_warn_steps` is rejected.
3. Threshold fields reject non-positive values.

Add middleware tests:

1. New, substantial tool output resets `consecutive_no_progress_steps`.
2. Repeated empty or duplicate results increments `consecutive_no_progress_steps`.
3. Warning-level stagnation queues exactly one change-strategy guidance message.
4. Queued guidance is injected in `wrap_model_call`, not directly in `after_model`.
5. Guidance is cleared in `after_agent`.
6. Non-interactive stagnation injects the non-interactive wrap-up guidance.
7. `max_tool_calls_without_answer` injects wrap-up guidance.
8. Hard-limit stagnation clears tool calls and raw provider metadata.
9. AI/tool message pairing remains valid after guidance injection.
10. LRU/run cleanup prevents unbounded in-memory tracking.
11. Recursion-pressure wrap-up guidance is queued when a visible `recursion_limit` is close to the observed model-turn count.
12. Recursion-pressure guidance is not queued when no `recursion_limit` is visible.

Add integration-style chain tests:

1. `DynamicLoopControllerMiddleware` is present when `dynamic_loop.enabled=true`.
2. It is omitted when `dynamic_loop.enabled=false`.
3. It is registered before `LoopDetectionMiddleware`.
4. It is registered before `ClarificationMiddleware`.

## Rollout

Default the feature to enabled, but keep thresholds conservative:

- It should first guide strategy before forcing termination.
- Hard stop should require multiple consecutive no-progress steps.
- Existing `LoopDetectionMiddleware` remains enabled and unchanged.

If initial review finds the default too interventionist, the fallback is to keep the config present but default `enabled=false` for one release. The code shape remains the same either way.

## Risks and Mitigations

### False Positive Stagnation

Some legitimate tasks involve many small tool outputs. Mitigation: count new paths, URLs, commands, artifacts, and fingerprints as progress even when text is short.

### Overlap with ToolProgressMiddleware

ToolProgressMiddleware already detects tool-result problems. Mitigation: keep this controller at run-level only and avoid duplicating per-tool block state.

### Provider Message Validation

Improper guidance insertion can break tool-call pairing. Mitigation: inject only in `wrap_model_call` and hard-stop only by clearing tool-call metadata.

### Hidden Cost of State Tracking

Long-lived processes may accumulate state. Mitigation: LRU cap tracked `(thread_id, run_id)` entries and clear current run state in `after_agent`.

### Premature Final Answers

Hard stops can produce incomplete answers. Mitigation: conservative default thresholds, explicit uncertainty wording in guidance, and continued `recursion_limit` as the ultimate safety bound.

## Acceptance Criteria

| Requirement | Authoritative Evidence |
|---|---|
| `DynamicLoopConfig` defaults and threshold validation exist | `backend/tests/test_dynamic_loop_config.py` passes |
| Every `dynamic_loop` config field is represented in behavior or tests | `backend/tests/test_dynamic_loop_config.py` and `backend/tests/test_dynamic_loop_controller_middleware.py` cover all fields |
| Lead-agent chain includes the controller when enabled and omits it when disabled | `backend/tests/test_lead_agent_model_resolution.py` passes |
| Controller is before `LoopDetectionMiddleware` | `backend/tests/test_lead_agent_model_resolution.py` passes |
| Stagnation increments on no-progress tool results and resets on new information | `backend/tests/test_dynamic_loop_controller_middleware.py` passes |
| Strategy guidance is injected from `wrap_model_call` after tool results | `backend/tests/test_dynamic_loop_controller_middleware.py` passes |
| Non-interactive stalled runs are guided to complete without clarification | `backend/tests/test_dynamic_loop_controller_middleware.py` passes |
| Tool-call budget pressure queues wrap-up guidance | `backend/tests/test_dynamic_loop_controller_middleware.py` passes |
| Recursion pressure queues wrap-up guidance only when `recursion_limit` is visible | `backend/tests/test_dynamic_loop_controller_middleware.py` passes |
| Hard-stop clears structured and raw provider tool metadata | `backend/tests/test_dynamic_loop_controller_middleware.py` passes |
| Existing repeated-call loop and token-budget guards keep passing | `backend/tests/test_loop_detection_middleware.py` and `backend/tests/test_token_budget_middleware.py` pass |
| Config schema bump is present | `config.example.yaml` has `config_version: 20` and a documented `dynamic_loop` block |
| Architecture docs describe role and boundaries | `README.md` and `backend/AGENTS.md` include dynamic loop control notes |

## Self-Acceptance Procedure

Before the feature can be considered complete, run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_dynamic_loop_config.py tests/test_dynamic_loop_controller_middleware.py tests/test_lead_agent_model_resolution.py tests/test_loop_detection_middleware.py tests/test_token_budget_middleware.py -q
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_config_version.py tests/test_app_config_name_indexes.py tests/test_app_config_reload.py tests/test_reload_boundary.py tests/test_harness_boundary.py tests/test_dynamic_loop_config.py -q
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check packages/harness/deerflow/config/dynamic_loop_config.py packages/harness/deerflow/config/app_config.py packages/harness/deerflow/config/__init__.py packages/harness/deerflow/agents/middlewares/dynamic_loop_controller_middleware.py packages/harness/deerflow/agents/lead_agent/agent.py tests/test_dynamic_loop_config.py tests/test_dynamic_loop_controller_middleware.py tests/test_lead_agent_model_resolution.py
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff format --check packages/harness/deerflow/config/dynamic_loop_config.py packages/harness/deerflow/config/app_config.py packages/harness/deerflow/config/__init__.py packages/harness/deerflow/agents/middlewares/dynamic_loop_controller_middleware.py packages/harness/deerflow/agents/lead_agent/agent.py tests/test_dynamic_loop_config.py tests/test_dynamic_loop_controller_middleware.py tests/test_lead_agent_model_resolution.py
```

Then inspect `git status --short --branch` and `git diff --stat` to confirm the remaining diff is limited to the RFC, plan, middleware/config implementation, tests, and docs listed in this RFC.
