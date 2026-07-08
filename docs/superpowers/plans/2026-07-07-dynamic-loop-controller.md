# Dynamic Loop Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable lead-agent `DynamicLoopControllerMiddleware` that detects run-level stagnation, nudges strategy changes, and forces safe wrap-up after repeated no-progress rounds without replacing the existing `create_agent(...)` loop.

**Architecture:** Keep DeerFlow's current LangGraph ReAct execution path. Add a Pydantic `DynamicLoopConfig`, a focused lead-only middleware with bounded run-scoped in-memory state, and register it before `LoopDetectionMiddleware`. The controller observes completed `ToolMessage`s at model-call boundaries and injects guidance only from `wrap_model_call`, preserving provider tool-call pairing.

**Tech Stack:** Python, Pydantic config models, LangChain `AgentMiddleware`, LangChain `ModelRequest`, pytest, ruff, existing DeerFlow middleware test patterns.

## Global Constraints

- Keep the existing `create_agent(...)` lead-agent loop as the execution engine.
- Do not add an explicit `StateGraph` in P0.
- Do not add evaluator-model calls.
- Do not change Gateway run lifecycle, `RunManager`, `StreamBridge`, persistence schemas, frontend UI, or `ThreadState`.
- Apply P0 to the lead agent only; subagent dynamic-loop support is out of scope.
- Preserve provider tool-call pairing: never insert a message between `AIMessage(tool_calls=...)` and matching `ToolMessage`s.
- Queue guidance and inject it only from `wrap_model_call`.
- Hard stop must clear structured `tool_calls`, raw provider metadata in `additional_kwargs`, legacy `function_call`, and `finish_reason="tool_calls"`.
- Every `dynamic_loop` config field must have implemented, tested behavior.
- Update `config.example.yaml` and bump `config_version` because this is a config schema change.
- Update `README.md` and `backend/AGENTS.md` because this adds a user-facing config block and a new lead-agent middleware.

---

## File Structure

- Create `backend/packages/harness/deerflow/config/dynamic_loop_config.py`
  - Defines `DynamicLoopConfig`, defaults, and threshold validation.
- Modify `backend/packages/harness/deerflow/config/app_config.py`
  - Adds `AppConfig.dynamic_loop`.
- Modify `backend/packages/harness/deerflow/config/__init__.py`
  - Re-exports `DynamicLoopConfig`.
- Create `backend/packages/harness/deerflow/agents/middlewares/dynamic_loop_controller_middleware.py`
  - Implements run-scoped progress tracking, pending guidance, recursion-pressure wrap-up, request injection, hard-stop mutation, and cleanup.
- Modify `backend/packages/harness/deerflow/agents/lead_agent/agent.py`
  - Registers `DynamicLoopControllerMiddleware` before `LoopDetectionMiddleware` when enabled.
- Create `backend/tests/test_dynamic_loop_config.py`
  - Tests config defaults and validation.
- Create `backend/tests/test_dynamic_loop_controller_middleware.py`
  - Tests progress/no-progress tracking, guidance injection, non-interactive behavior, tool-call budget pressure, recursion pressure, hard stop, and cleanup.
- Modify `backend/tests/test_lead_agent_model_resolution.py`
  - Tests lead-agent chain inclusion, disabled mode, config propagation, and placement before `LoopDetectionMiddleware`.
- Modify `config.example.yaml`
  - Bumps `config_version` to `20` and documents `dynamic_loop`.
- Modify `README.md`
  - Adds a concise user-facing description under Context Engineering.
- Modify `backend/AGENTS.md`
  - Documents middleware role, placement, and division of labor with `ToolProgressMiddleware` and `LoopDetectionMiddleware`.

---

### Task 1: Add Dynamic Loop Config

**Files:**
- Create: `backend/packages/harness/deerflow/config/dynamic_loop_config.py`
- Modify: `backend/packages/harness/deerflow/config/app_config.py`
- Modify: `backend/packages/harness/deerflow/config/__init__.py`
- Create: `backend/tests/test_dynamic_loop_config.py`

**Interfaces:**
- Produces: `DynamicLoopConfig`
- Consumed by: `DynamicLoopControllerMiddleware.from_config(config: DynamicLoopConfig)`

- [x] **Step 1: Write failing config tests**

Tests must assert:

- defaults: `enabled=True`, `stagnation_warn_steps=3`, `stagnation_hard_limit=6`, `max_tool_calls_without_answer=40`, `wrap_up_when_recursion_remaining=8`, `min_new_info_chars=120`, `force_wrap_up_on_non_interactive_stagnation=True`;
- `stagnation_hard_limit < stagnation_warn_steps` raises `ValidationError`;
- non-positive threshold fields raise `ValidationError`.

- [x] **Step 2: Verify RED**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_dynamic_loop_config.py -q
```

Expected before implementation: FAIL with `ModuleNotFoundError: No module named 'deerflow.config.dynamic_loop_config'`.

- [x] **Step 3: Implement config**

Create `DynamicLoopConfig` with Pydantic `Field(..., ge=1)` validation for numeric thresholds and an `after` validator enforcing `stagnation_hard_limit >= stagnation_warn_steps`.

- [x] **Step 4: Wire AppConfig and exports**

Add `dynamic_loop: DynamicLoopConfig = Field(default_factory=DynamicLoopConfig, description="Run-level dynamic loop controller middleware configuration")` to `AppConfig`.

- [x] **Step 5: Verify GREEN**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_dynamic_loop_config.py -q
```

Expected after implementation: PASS.

---

### Task 2: Implement DynamicLoopControllerMiddleware

**Files:**
- Create: `backend/packages/harness/deerflow/agents/middlewares/dynamic_loop_controller_middleware.py`
- Create: `backend/tests/test_dynamic_loop_controller_middleware.py`

**Interfaces:**
- Consumes: `DynamicLoopConfig`
- Produces:
  - `DynamicLoopControllerMiddleware.from_config(config: DynamicLoopConfig) -> DynamicLoopControllerMiddleware`
  - `DynamicLoopControllerMiddleware.reset(thread_id: str | None = None) -> None`
  - guidance constants `_CHANGE_STRATEGY_GUIDANCE`, `_WRAP_UP_GUIDANCE`, `_NON_INTERACTIVE_GUIDANCE`, `_FORCED_WRAP_UP_MSG`

- [x] **Step 1: Write failing middleware tests**

Tests must cover:

- substantial new tool output resets no-progress count;
- repeated no-progress tool results queue change-strategy guidance once;
- guidance is injected from `wrap_model_call`;
- non-interactive stagnation queues non-interactive wrap-up guidance;
- tool-call budget pressure queues wrap-up guidance;
- visible `runtime.config["recursion_limit"]` close to observed model turns queues wrap-up guidance;
- absent recursion limit does not queue recursion-pressure guidance;
- hard stop strips structured `tool_calls`, raw provider `tool_calls`, legacy `function_call`, and changes `finish_reason="tool_calls"` to `"stop"`;
- `after_agent` clears run state and pending guidance.

- [x] **Step 2: Verify RED**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_dynamic_loop_controller_middleware.py -q
```

Expected before implementation: FAIL with `ModuleNotFoundError: No module named 'deerflow.agents.middlewares.dynamic_loop_controller_middleware'`. For the later recursion-pressure addition, expected RED was a failed assertion that no wrap-up guidance was queued.

- [x] **Step 3: Implement middleware**

Implementation requirements:

- maintain bounded state keyed by `(thread_id, run_id)`;
- track `observed_model_turns`, `tool_calls_since_answer`, `consecutive_no_progress_steps`, `guidance_sent`, recent content fingerprints, and seen tool messages;
- treat new substantial text, new salient references, or success/partial-success metadata as progress;
- treat empty, duplicate, or recoverable error results as no progress;
- queue guidance once per run per guidance type;
- read recursion pressure only from a visible positive integer `runtime.config["recursion_limit"]`;
- skip recursion-pressure checks if no valid limit is visible;
- inject guidance as a hidden `HumanMessage` only in `wrap_model_call`;
- hard stop by mutating only the current `AIMessage` into a plain assistant response.

- [x] **Step 4: Verify GREEN**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_dynamic_loop_controller_middleware.py -q
```

Expected after implementation: PASS.

---

### Task 3: Register Lead-Agent Middleware

**Files:**
- Modify: `backend/packages/harness/deerflow/agents/lead_agent/agent.py`
- Modify: `backend/tests/test_lead_agent_model_resolution.py`

**Interfaces:**
- Consumes: `AppConfig.dynamic_loop`
- Produces: lead-agent middleware chain includes `DynamicLoopControllerMiddleware` when enabled

- [x] **Step 1: Write failing chain tests**

Tests must assert:

- middleware is present by default;
- middleware is omitted when `DynamicLoopConfig(enabled=False)`;
- middleware receives config values from `AppConfig.dynamic_loop`;
- middleware index is lower than `LoopDetectionMiddleware`.

- [x] **Step 2: Verify RED**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_lead_agent_model_resolution.py -q
```

Expected before registration: FAIL because no `DynamicLoopControllerMiddleware` appears in the chain.

- [x] **Step 3: Register middleware**

Add import:

```python
from deerflow.agents.middlewares.dynamic_loop_controller_middleware import DynamicLoopControllerMiddleware
```

Add registration immediately before `LoopDetectionMiddleware`:

```python
dynamic_loop_config = resolved_app_config.dynamic_loop
if dynamic_loop_config.enabled:
    middlewares.append(DynamicLoopControllerMiddleware.from_config(dynamic_loop_config))
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_lead_agent_model_resolution.py -q
```

Expected after registration: PASS.

---

### Task 4: Update User and Architecture Docs

**Files:**
- Modify: `config.example.yaml`
- Modify: `README.md`
- Modify: `backend/AGENTS.md`

**Interfaces:**
- Consumes: implemented `dynamic_loop` config and middleware chain
- Produces: user-facing config reference and architecture guidance

- [x] **Step 1: Update config example**

Set `config_version: 20` and add:

```yaml
dynamic_loop:
  enabled: true
  stagnation_warn_steps: 3
  stagnation_hard_limit: 6
  max_tool_calls_without_answer: 40
  wrap_up_when_recursion_remaining: 8
  min_new_info_chars: 120
  force_wrap_up_on_non_interactive_stagnation: true
```

- [x] **Step 2: Update README**

Add a concise Context Engineering note explaining dynamic loop control and the `dynamic_loop` config block.

- [x] **Step 3: Update backend AGENTS guide**

Document `DynamicLoopControllerMiddleware` in the lead-only chain before `LoopDetectionMiddleware`, and explain that:

- `ToolProgressMiddleware` is per-tool result quality;
- `DynamicLoopControllerMiddleware` is run-level stagnation and wrap-up coordination;
- `LoopDetectionMiddleware` is repeated call-pattern safety.

---

### Task 5: Final Verification and Self-Acceptance

**Files:**
- Verify every file listed in Tasks 1-4 plus `docs/superpowers/specs/2026-07-07-dynamic-loop-controller-design.md`.

**Interfaces:**
- Consumes: completed implementation and docs
- Produces: evidence that P0 is complete

- [ ] **Step 1: Run focused behavior tests**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_dynamic_loop_config.py tests/test_dynamic_loop_controller_middleware.py tests/test_lead_agent_model_resolution.py tests/test_loop_detection_middleware.py tests/test_token_budget_middleware.py -q
```

Expected: PASS.

- [ ] **Step 2: Run config and boundary regression tests**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run pytest tests/test_config_version.py tests/test_app_config_name_indexes.py tests/test_app_config_reload.py tests/test_reload_boundary.py tests/test_harness_boundary.py tests/test_dynamic_loop_config.py -q
```

Expected: PASS.

- [ ] **Step 3: Run ruff checks**

Run:

```bash
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff check packages/harness/deerflow/config/dynamic_loop_config.py packages/harness/deerflow/config/app_config.py packages/harness/deerflow/config/__init__.py packages/harness/deerflow/agents/middlewares/dynamic_loop_controller_middleware.py packages/harness/deerflow/agents/lead_agent/agent.py tests/test_dynamic_loop_config.py tests/test_dynamic_loop_controller_middleware.py tests/test_lead_agent_model_resolution.py
cd backend && UV_CACHE_DIR=/private/tmp/uv-cache uv run ruff format --check packages/harness/deerflow/config/dynamic_loop_config.py packages/harness/deerflow/config/app_config.py packages/harness/deerflow/config/__init__.py packages/harness/deerflow/agents/middlewares/dynamic_loop_controller_middleware.py packages/harness/deerflow/agents/lead_agent/agent.py tests/test_dynamic_loop_config.py tests/test_dynamic_loop_controller_middleware.py tests/test_lead_agent_model_resolution.py
```

Expected: PASS.

- [ ] **Step 4: Run plan/spec consistency scan**

Run:

```bash
rg -n "TBD|TODO|implement later|wrap_tool_call|ToolCallRequest|last_new_info_fingerprint|\\bmodel_turns\\b|DynamicLoopControllerMiddleware\\(" docs/superpowers/specs/2026-07-07-dynamic-loop-controller-design.md docs/superpowers/plans/2026-07-07-dynamic-loop-controller.md | rg -v "rg -n"
```

Expected: no output after filtering out the scan command itself.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git status --short --branch
git diff --stat
```

Expected: remaining diff is limited to dynamic-loop RFC, plan, config, middleware, tests, config example, README, and backend AGENTS docs. Pre-existing unrelated untracked files must be called out explicitly rather than silently included.

- [ ] **Step 6: Self-acceptance audit**

Check every RFC acceptance row:

- `dynamic_loop.enabled` controls lead-agent registration.
- `stagnation_warn_steps`, `stagnation_hard_limit`, `max_tool_calls_without_answer`, `wrap_up_when_recursion_remaining`, `min_new_info_chars`, and `force_wrap_up_on_non_interactive_stagnation` each affect tested behavior.
- Guidance is injected only from `wrap_model_call`.
- Hard stop clears structured and raw provider tool-call metadata.
- Existing loop detection and token-budget tests still pass.
- README, `backend/AGENTS.md`, and `config.example.yaml` document the user-facing and architecture-facing changes.

Expected: every bullet is proven by test output, doc diff, or inspected code path.

---

## Self-Review

- Spec coverage: This plan covers config, middleware, lead-agent registration, config example, README, backend architecture docs, tests, and final verification. It explicitly excludes StateGraph, evaluator LLM calls, frontend changes, Gateway lifecycle changes, persistence changes, `ThreadState`, and subagent support.
- Placeholder scan: The plan contains no unresolved placeholder markers or references to future implementation as part of P0 completion.
- Type consistency: The plan uses `DynamicLoopConfig`, `DynamicLoopControllerMiddleware`, `DynamicLoopRunState`, and constants with the same names as the implementation. Hook signatures match the P0 implementation: `wrap_model_call`, `awrap_model_call`, `after_model`, `aafter_model`, `after_agent`, and `aafter_agent`.
