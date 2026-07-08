# DeerFlow Long-Horizon Skill-First Strategy Design

**Date**: 2026-07-08
**Status**: Draft for user review
**Scope**: Long-horizon task support through dynamic loop control, progress tracking, and curated DeerFlow skill packs

---

## Decision Summary

DeerFlow should **not** implement a full explicit `StateGraph` dynamic loop as the next mainline architecture change.

Instead, DeerFlow should take a skill-first path:

1. Keep the existing `create_agent(...)` ReAct loop as the default execution engine.
2. Add `DynamicLoopControllerMiddleware` for run-level stagnation and wrap-up control.
3. Add a lightweight progress ledger before any explicit long-horizon graph.
4. Curate and adapt mainstream workflow skills into DeerFlow's `skills/public` or `skills/custom` ecosystem.
5. Use long-horizon benchmark tasks to decide, after evidence is available, whether an opt-in `long_horizon_agent` `StateGraph` is justified.

The goal is to improve long-task behavior without creating a second runtime stack.

## Background

Recent DeerFlow issue analysis points to a consistent pattern:

- agents loop or over-explore because they cannot tell when a task phase is done;
- loop detection can confuse legitimate long workflows with runaway behavior;
- subagents and non-interactive runs need clearer budget and stop semantics;
- benchmark-like tasks need better task decomposition, progress tracking, verification, and summarization discipline.

Those problems are not primarily caused by the absence of an explicit `StateGraph`. They are caused by weak progress semantics and missing task-level operating procedures.

Skills are the right near-term lever because they can encode task-specific operating procedures without changing the runtime graph.

## Goals

1. Improve long-horizon task success for research, coding, benchmark, scheduled, and repo-analysis tasks.
2. Keep the default lead agent architecture stable.
3. Reuse DeerFlow's existing skill system instead of hardcoding every workflow into Python graph nodes.
4. Introduce mainstream workflow skills in a controlled, auditable way.
5. Produce benchmark evidence before deciding whether an explicit `StateGraph` agent is worth building.

## Non-Goals

This design does not include:

1. Replacing `lead_agent` with a custom `StateGraph`.
2. Adding a new `long_horizon_agent` implementation immediately.
3. Installing arbitrary third-party skills directly into `skills/public`.
4. Copying Codex/Superpowers skills verbatim into DeerFlow without adaptation.
5. Adding new frontend UI.
6. Changing Gateway run lifecycle, `RunManager`, `StreamBridge`, or persistence schemas.

## Recommended Architecture

### Layer 1: Existing ReAct Runtime

Keep the current lead agent:

```text
user input
  -> create_agent(...)
  -> model/tool loop
  -> middleware guards
  -> final answer
```

This remains the default for normal chats, custom agents, channels, and scheduled executions.

### Layer 2: Dynamic Loop Controller

Implement the existing `DynamicLoopControllerMiddleware` design:

- detect run-level stagnation;
- inject change-strategy or wrap-up guidance;
- handle non-interactive stalled runs;
- force finalization after repeated no-progress rounds.

This solves the broad class of "keeps searching / keeps reading / keeps retrying" behavior without changing the graph.

### Layer 3: Progress Ledger

Add a lightweight progress ledger after the dynamic loop controller lands.

The ledger should record:

```text
phase
subtask/topic
tool calls
new evidence keys
artifacts changed
files modified
verification status
stagnation count
stop reason
```

The first version can be in-memory and middleware-owned. Persisted `ThreadState` should be considered only after tests prove the data must survive summarization or restarts.

### Layer 4: Curated Skill Packs

Use skills to encode long-horizon workflows:

- how to decompose the task;
- when to research;
- when to stop researching;
- when to modify source;
- when to verify;
- how to summarize uncertainty;
- how to avoid repeated scaffolding or search escalation.

This keeps policy close to the task domain instead of forcing every workflow into one Python state machine.

### Layer 5: Optional Future StateGraph

Only after the above layers have benchmark data should DeerFlow consider an opt-in explicit graph:

```text
intake -> plan -> execute_step -> assess_progress -> route -> verify -> synthesize
```

This future graph should be a separate profile, not a replacement for `lead_agent`.

## Skill Pack Strategy

There are two different skill ecosystems in play:

1. **DeerFlow runtime skills** under `skills/public` and `skills/custom`.
2. **Codex/Superpowers development skills** under the developer's Codex skill directory.

They should not be mixed blindly. Codex/Superpowers skills often contain tool-specific instructions for Codex, git worktrees, commits, browser plugins, or local orchestration. DeerFlow skills must be rewritten for DeerFlow's tools, sandbox, MCP, and runtime constraints.

## Existing DeerFlow Skills to Promote

The repository already has several useful public skills:

| Skill | Recommended role |
|---|---|
| `deep-research` | General web research methodology and synthesis discipline |
| `github-deep-research` | Repository and issue research reports |
| `data-analysis` | Data inspection, transformation, and explanation |
| `chart-visualization` | Structured visual outputs from analyzed data |
| `code-documentation` | Documentation generation and code explanation |
| `frontend-design` | Product/UI design workflows |
| `systematic-literature-review` | Academic-style multi-source review |
| `academic-paper-review` | Focused paper analysis |
| `consulting-analysis` | Business/strategy analysis |
| `skill-creator` | Creating or refining DeerFlow skills |
| `find-skills` | Discovering external skills from the open skill ecosystem |

These should be used as the initial long-horizon benchmark set before importing many new skills.

## Recommended New or Adapted Skills

### 1. `repo-issue-fix`

Purpose: fix a GitHub issue or local bug report end to end.

Workflow:

```text
read issue -> map repo -> reproduce -> write failing test -> implement -> verify -> summarize
```

Source inspiration:

- Superpowers `systematic-debugging`
- Superpowers `test-driven-development`
- Superpowers `verification-before-completion`
- SWE-bench failure patterns from DeerFlow issue analysis

Do not copy those skills verbatim. Convert them into DeerFlow instructions that use DeerFlow tools and avoid Codex-only assumptions.

### 2. `repo-map`

Purpose: build a compact mental model of a repository before changing code.

Workflow:

```text
inspect tree -> identify modules -> read local AGENTS.md guides -> map entrypoints -> summarize ownership boundaries
```

This can be inspired by gitingest-like or "repo digest" workflows, but should stay local-first and avoid requiring an external SaaS.

### 3. `long-research-report`

Purpose: handle multi-topic research without endless search/fetch loops.

Workflow:

```text
topic list -> per-topic search budget -> evidence table -> sufficiency check -> final synthesis
```

This should extend `deep-research` rather than replace it.

### 4. `scheduled-noninteractive-run`

Purpose: make unattended scheduled tasks reliable.

Workflow:

```text
state assumptions -> avoid clarification -> use available context -> produce result with uncertainty
```

This directly supports scheduled tasks where `context.non_interactive=true`.

### 5. `benchmark-runner`

Purpose: run and analyze long-horizon benchmarks such as SWE-bench-like tasks.

Workflow:

```text
load case -> run agent -> collect trajectory -> classify failure -> propose intervention
```

This skill should be used to decide whether a future explicit `StateGraph` profile is justified.

### 6. `pr-review-and-fix`

Purpose: inspect review feedback, decide which comments are actionable, implement fixes, and verify.

Source inspiration:

- Superpowers `receiving-code-review`
- Superpowers `requesting-code-review`
- GitHub PR review workflows

### 7. `release-readiness`

Purpose: inspect changelog, open PRs, tests, docs, and config migrations before release.

Workflow:

```text
collect repo state -> inspect CI/test signals -> check docs/config migrations -> produce release risk report
```

### 8. `skill-import-adapter`

Purpose: adapt external skills into DeerFlow format safely.

Workflow:

```text
read external skill -> identify tool assumptions -> remove incompatible instructions -> map to DeerFlow tools -> add trigger tests
```

This should be the gate for any Superpowers, skills.sh, Vercel, Composio, or community skill being pulled into DeerFlow.

## External Skill Sources to Consider

Recommended sources:

1. **Superpowers skills** already used by the development agent:
   - systematic debugging
   - test-driven development
   - verification before completion
   - writing plans
   - receiving/requesting code review
   - using git worktrees
   - finishing a development branch

2. **Open skills ecosystem via `find-skills` / Skills CLI**:
   - React/Next.js best practices
   - testing and E2E workflows
   - deployment and CI/CD workflows
   - documentation and changelog workflows
   - accessibility and design-system workflows

3. **Repo-digestion workflows**:
   - gitingest-style repository summarization
   - issue triage
   - architecture map generation

These should be treated as inspiration or inputs, not trusted project code.

## Skill Import Policy

Any imported skill must pass these gates:

1. **License check**: confirm the source license permits reuse.
2. **Tool compatibility check**: remove assumptions about unavailable tools.
3. **Security check**: no credential exfiltration, destructive defaults, hidden network actions, or untrusted shell snippets.
4. **Runtime fit**: map file operations to DeerFlow sandbox conventions.
5. **Trigger quality**: description should be specific enough to trigger when useful and not trigger for unrelated tasks.
6. **Evaluation prompts**: each skill must include 3-5 test prompts for long-horizon behavior.
7. **Documentation**: public skills require `README.md` or `AGENTS.md` updates only when user-facing behavior changes.

## Benchmark Tasks

Use these tasks to test the skill-first approach:

### GitHub Issue Fix

Prompt:

```text
Given this issue URL, inspect the repository, reproduce the bug, write a failing test, implement the fix, run verification, and summarize the patch.
```

Expected skills:

- `repo-issue-fix`
- `repo-map`
- `code-documentation`

Success signals:

- source file changes happen after focused exploration;
- tests are run;
- final answer includes verification evidence.

### Multi-Topic Research Report

Prompt:

```text
Research 10 DeerFlow runtime problems one topic at a time. For each topic, search once, fetch only the best source, decide whether evidence is enough, then move on.
```

Expected skills:

- `long-research-report`
- `deep-research`
- `github-deep-research`

Success signals:

- each topic has bounded search;
- no search/fetch alternation loop;
- final synthesis covers all topics.

### Scheduled Non-Interactive Summary

Prompt:

```text
Run a daily project summary with no ability to ask clarification. Use available repo, issue, and run context; state assumptions and produce a concise report.
```

Expected skills:

- `scheduled-noninteractive-run`
- `release-readiness`

Success signals:

- no clarification request;
- assumptions are explicit;
- output is complete despite missing context.

### SWE-Bench-Style Debugging

Prompt:

```text
Fix this failing behavior in a large Python repo. Do not create repeated throwaway scripts unless necessary. Prefer reading existing tests and writing one focused regression test.
```

Expected skills:

- `repo-issue-fix`
- `repo-map`
- `benchmark-runner`

Success signals:

- bounded repository exploration;
- low scaffold-to-source ratio;
- verification run before final answer.

## Why This Comes Before StateGraph

An explicit `StateGraph` can enforce stages, but it cannot by itself define good progress semantics.

Without good skills and progress tracking, a graph can still loop inside an `execute` node, summarize too early, or route incorrectly. Skills provide domain-specific procedures that can be reused inside graph nodes if benchmark evidence shows an explicit long-horizon graph is necessary.

Therefore the right order is:

```text
DynamicLoopControllerMiddleware
  -> Progress ledger
  -> Curated skill packs
  -> Benchmarks
  -> Optional StateGraph profile
```

## Implementation Phases

### Phase 1: Finish Dynamic Loop Controller

Implement the existing dynamic loop controller spec and tests.

Deliverables:

- `DynamicLoopConfig`
- `DynamicLoopControllerMiddleware`
- config example update
- middleware chain tests

### Phase 2: Add Progress Ledger

Add lightweight progress tracking for long-horizon runs.

Deliverables:

- in-memory run progress object;
- evidence keys;
- artifact/file modification counters;
- stagnation counters;
- stop reason tracking.

### Phase 3: Curate Initial Skill Pack

Create or adapt:

- `repo-issue-fix`
- `repo-map`
- `long-research-report`
- `scheduled-noninteractive-run`
- `benchmark-runner`
- `pr-review-and-fix`
- `release-readiness`
- `skill-import-adapter`

Deliverables:

- new `skills/public` or `skills/custom` entries;
- test prompts for each;
- short benchmark tasks.

### Phase 4: Benchmark and Decide on StateGraph

Run the benchmark tasks with:

1. baseline current agent;
2. dynamic loop controller only;
3. dynamic loop controller + curated skills;
4. optional experimental explicit graph prototype if the first three still fail.

Decision rule:

- If curated skills + progress ledger solve the main failure cases, do not build StateGraph.
- If failures are mostly phase-ordering failures that middleware cannot enforce, design an opt-in `long_horizon_agent`.

## Acceptance Criteria

1. The default lead agent remains `create_agent(...)` based.
2. The project has a clear policy for adapting external skills into DeerFlow.
3. At least five long-horizon DeerFlow skills are selected or drafted before any explicit graph work.
4. Benchmark tasks cover coding, research, scheduled non-interactive work, and release/project analysis.
5. StateGraph remains a data-driven future option, not the immediate implementation path.

## Open Questions

1. Should curated imported skills live in `skills/public` or `skills/custom` first?
2. Should DeerFlow expose a skill marketplace/import UI, or keep imports as developer-managed files?
3. Should progress ledger data eventually persist in `ThreadState`, or remain middleware-local?
4. Which benchmark task should be the release gate for declaring long-horizon support improved?
