# What an agent loop is, and when you want one

## The shape

A single model call is a function: text in, text out. An agent is that call
placed inside a loop, where the output can be a request to *do something* and
the result of doing it becomes part of the next input.

```
  task ──▶ model ──▶ tool call ──▶ tool runs ──▶ result ──▶ model ──▶ ...
             ▲                                                │
             └────────────────────────────────────────────────┘
```

That is the whole idea. Everything else — planning, checking, budgets, guards —
exists because this loop, written naively, does not stop and does not know
whether it is working.

## When a loop is the right tool

Use one when **you cannot write down the steps in advance**, because what to do
next depends on what the previous step returned.

| Situation | Reach for |
|---|---|
| Fixed sequence of known steps | A script. It is faster, cheaper, and testable. |
| One question, context you already have | A single model call. |
| One question, context you must retrieve | RAG — see [01-rag](https://github.com/hamzadaniyal7/01-rag). |
| Unknown number of steps, each depending on the last | An agent loop. |
| Steps are known but the data is messy | A script that calls a model per item. Still not a loop. |

The honest version of this table: most things people build agents for are
scripts with a model call in the middle. A loop costs you determinism, a
predictable bill, and the ability to reason about what the system will do. Pay
that only when branching on observations is genuinely required.

## Why the naive loop fails

Write the loop above in twenty lines and run it on a real task. Four things
happen, in roughly this order:

**1. It does not stop.** The model keeps finding one more thing to check. There
is no step at which "enough" is defined, so the loop runs until something else
kills it — a timeout, a rate limit, your patience.

**2. It repeats itself.** A tool returns nothing useful; the model's most
plausible next action is the same call, because from inside the transcript that
call still looks like the right idea. Nothing in the loop notices that step 7
and step 8 and step 9 were identical.

**3. It reports success it did not achieve.** Asked afterwards whether it
completed the task, the model says yes. It is the same model that chose the
actions, reading the same transcript, and it has no independent evidence.

**4. It gets expensive in a way that surprises you.** Each step re-sends every
previous step. Cost is quadratic in steps, not linear, and the largest thing in
the transcript is tool output.

Each of those has a mechanism in this project aimed squarely at it:

| Failure | Mechanism | Where |
|---|---|---|
| Does not stop | Three budgets, checked before every step | `budget.py` |
| Repeats itself | Loop guard on call signatures and stalled verdicts | `loopguard.py` |
| Claims false success | A checker that is not the agent | `checker.py` |
| Costs more than expected | Observation caps, compaction, prompt caching | `context.py`, `tools/base.py` |

## Plan, act, check

The loop in this project has three phases rather than one.

**Plan** turns the task into ordered steps, each with a success criterion — an
observable fact that will be true when the step is done. The criterion is the
part that matters. Without it there is nothing to check against, and "did that
help?" has no answer.

**Act** is the ordinary loop: the model picks a tool, the tool runs, the result
comes back.

**Check** rules on what just happened. Cheap rules first (did the tool error?
return nothing? return exactly what it returned last time?), and only when
those are inconclusive, a model — a different, cheaper one — asked whether the
success criterion is now met.

The value is not that checking makes the agent smarter. It does not. The value
is that a run which is going nowhere **says so**, early, while there is still
budget left to do something about it, instead of producing a confident summary
of nine steps of nothing.

## What this project is not

It is not a framework. There is no plugin system, no DSL, no agent-of-agents.
It is about 4,500 lines of Python you can read in an afternoon, and the point is
that every mechanism above is visible rather than buried in a dependency.

It is also not a security boundary against a determined adversary. The sandbox
and the shell allowlist stop an agent making an expensive mistake. They do not
stop a hostile task description. See
[05-decisions.md](05-decisions.md#adr-3-the-sandbox-is-a-mistake-boundary-not-a-security-boundary).

Next: [the architecture](02-architecture.md).
