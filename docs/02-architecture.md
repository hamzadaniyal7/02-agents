# Architecture

## Module map

26 modules, grouped by what they are responsible for.

```
src/actcheck/
├── agent.py        the loop: plan -> act -> check -> recover
├── planner.py      task -> plan; failing plan -> revised plan
├── checker.py      did that step help? (rules, then a model)
├── loopguard.py    repetition and stall detection
├── budget.py       step / token / wall-clock ceilings
├── context.py      the transcript, and compaction
├── trace.py        append-only JSONL record of a run
├── prompts.py      every prompt, in one diffable place
├── types.py        the data model
├── config.py       one dataclass, three sources
├── errors.py       typed failures
├── evaluate.py     the eval harness
├── cli.py          the command line
├── tools/
│   ├── base.py     Tool contract + Registry (the only place tools run)
│   ├── sandbox.py  the workspace boundary
│   ├── fs.py       read_file, write_file, list_dir
│   ├── search.py   grep
│   ├── shell.py    run_command
│   └── finish.py   finish, give_up
└── llm/
    ├── base.py          the model contract
    ├── anthropic_llm.py tool use, caching, adaptive thinking
    ├── scripted.py      the offline stand-in
    └── replay.py        re-execute a recorded trace
```

## The data model

Everything that flows through a run is a plain dataclass in `types.py`:

```
Run ──▶ Plan ──▶ Step* ──▶ ToolCall ──▶ Observation ──▶ Verdict
```

* **`Plan`** — a goal and ordered `PlanStep`s, each with a `success_criterion`.
* **`Step`** — one trip round the loop. Holds the tool call, what came back,
  and what the checker made of it.
* **`ToolCall`** — name plus arguments, with a `signature()` that is stable
  across key reordering. The loop guard is built on that stability.
* **`Observation`** — what a tool returned. `ok=False` is a normal outcome, not
  an exception: a failure is information the model needs.
* **`Verdict`** — the checker's ruling, and who made it (`programmatic`,
  `model`, or `guard`).
* **`RunResult`** — a terminal `status` plus a `reason`. Every way a run can
  end is named; there is no "it just stopped".

## One trip round the loop

```
                   ┌──────────────────── transcript ◀───────────────┐
                   ▼                                                │
  budget check ──▶ model.respond(system, messages, tools)           │
       │                        │                                   │
       │ exhausted              ▼ tool_use                          │
       ▼                   loop guard ──── tripped ──▶ nudge ───────┤
  BUDGET_EXHAUSTED              │ clear                             │
                                ▼                                   │
                        registry.execute(call)                      │
                                │                                   │
                                ▼                                   │
                        checker.check(...)                          │
                                │                                   │
              ┌─────────────────┼─────────────────┐                 │
              ▼                 ▼                 ▼                 │
          GOAL_DONE         advanced          stalled               │
              │                 │                 │                 │
              ▼                 └─────────────────┼─────────────────┘
           SUCCESS                                ▼
                                        replan / nudge / STUCK
```

Every box writes to the trace as it happens.

## Where each decision lives

The design constraint throughout is that each decision has exactly one home.

| Decision | Lives in | Not in |
|---|---|---|
| Which tools exist | `tools/__init__.build_registry` | the agent |
| Whether a path is allowed | `tools/sandbox.Sandbox.resolve` | each tool |
| How a tool failure is reported | `tools/base.Registry.execute` | each tool |
| How big an observation may be | `Registry._cap` | each tool |
| Whether a step helped | `checker.Checker` | the agent's own opinion |
| When to stop | `budget.Budget` + `loopguard.LoopGuard` | the model |
| What the model is told | `prompts.py` | f-strings next to the logic |
| What is recorded | `trace.Trace` | log statements |

The payoff is in the tools: a new one implements `run()` and a schema, and
inherits sandboxing, validation, timing, truncation and error capture without
knowing those exist.

## The model boundary

`llm/base.py` defines two methods:

* `respond(system, messages, tools)` — the agent's own conversation.
* `complete(system, user, schema, purpose)` — single-turn, structured. Used by
  the planner and the checker.

Keeping them apart is what lets the checker run on a cheaper model *and* keeps
the checker's reasoning out of the context the agent sees.

Three providers implement it: `anthropic` (real), `scripted` (offline and
deterministic), and `replay` (re-executes a recorded trace). The whole test
suite and the entire eval harness run on the second one, which is why neither
needs an API key.

The transcript is carried in Anthropic's content-block shape rather than a
bespoke neutral format. That is a deliberate trade, written up in
[05-decisions.md](05-decisions.md#adr-5-the-transcript-uses-the-providers-message-shape).

Next: [the loop, stage by stage](03-the-loop.md).
