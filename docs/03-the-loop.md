# The loop, stage by stage

Every knob in `actcheck.toml`, what it does, and the failure it exists to
prevent.

---

## Plan

**`plan`, `plan_max_steps`, `max_replans`**

One model call up front turns the task into ordered steps, each with a
`success_criterion` — an observable fact that will be true when the step is
done. The planner prompt is explicit that "understand the codebase" is not a
criterion and "README.md has been read and its build command identified" is.

Planning is optional (`plan = false`) and the loop works without it. What you
lose is the thing the checker checks *against*: with no criterion, a verdict
can only be "the tool returned bytes", which is a much weaker signal.

**Replanning** is the part that earns its keep. A plan that cannot be revised
makes the first guess binding, and the agent spends its whole budget failing
the same way. A plan revised on every setback means nothing ever finishes. So
replanning fires only when the loop guard reports a stall, it is handed the
evidence of what actually went wrong, and `max_replans` caps it. Past the cap
the run ends as `STUCK` — a third rewrite of the plan is not a plan problem.

Steps already marked done are carried across a revision. Re-doing finished work
is the commonest way a replanning agent burns the budget it has left.

---

## Act

**`max_output_tokens`, `effort`, `thinking`**

The model is given the transcript and the tool schemas, and returns either
prose or tool calls. Tool calls are executed through the registry, which is the
only place tools run — so every observation is timed, size-capped and
error-wrapped identically.

**There is no retry-with-backoff around tool failures.** A tool that failed
because the model passed a bad path does not want retrying; it wants the model
to read the error. Retrying is the mechanism that turns one bad decision into
five identical ones.

**A model turn with no tool call** gets one nudge, then ends the run. Two silent
turns means it is not going to use the tools, and burning the budget to confirm
that helps nobody.

---

## Check

**`checker`, `checker_model`, `checker_effort`**

| Mode | What happens | Cost |
|---|---|---|
| `off` | Trust the agent's own say-so | free |
| `programmatic` | Rules over the observation | free |
| `model` | Ask a model against the criterion | one call per step |
| `both` *(default)* | Rules first, model only when inconclusive | one call per *interesting* step |

The programmatic rules are certain and blind to meaning:

* the tool errored → `no_progress`
* it returned nothing → `no_progress`
* it succeeded but found nothing (`No matches for…`) → `no_progress`
* it returned exactly what it returned last time → `no_progress`
* `finish` → `goal_done`; `give_up` → `blocked`

That last-but-one rule catches what the loop guard cannot: two *different*
calls returning identical content, e.g. reading one file by two paths.

The empty-result rule is worth dwelling on. A `grep` that matches nothing is a
success at the tool level and a non-event at the task level, and conflating the
two is how an agent racks up nine "successful" steps having learned nothing.

When the rules cannot tell — the tool worked, returned plausible content, and
the question is whether that content *satisfies the criterion* — a model is
asked. That is a judgement about meaning and nothing else will make it. It runs
on a cheaper model because grading one observation against one sentence is a far
smaller job than deciding what to do next.

The order matters: cheap-and-certain first. Inverting it roughly doubles the
cost of a run to learn things the rules already knew.

---

## Guard

**`repeat_threshold`, `no_progress_threshold`, `step_attempt_limit`**

Two detectors for two shapes of stuck.

**Repetition** — the same tool with the same arguments, back to back. Detected
exactly via `ToolCall.signature()`, which sorts keys so that reordering is not
mistaken for a new action.

**No progress** — different calls, none of which the checker credits. The
subtler one: every step is a new action, the agent looks busy, nothing moves.
Only a checker that rules on outcomes can see this, which is why `checker` is
not `off` by default.

What the guard does on firing matters as much as the detection. It does **not**
kill the run on first trip. It replaces the repeated call with a message in the
transcript telling the agent what it just did:

```
You have now called read_file with identical arguments 3 times in a row and
the result has not changed. It will not change on another attempt. Either use
a different tool or different arguments, or call give_up explaining what is
blocking you.
```

That converts a hidden pattern into visible evidence, and it is usually enough.
Only when the nudge fails does the run end as `STUCK` — which is an honest
outcome, and a better one than burning the whole budget to arrive at the same
place.

---

## Budget

**`max_steps`, `max_tokens`, `max_seconds`**

Three independent ceilings, because they fail for different reasons:

* **steps** — a loop making cheap calls and getting nowhere is invisible to a
  token budget.
* **tokens** — long tool outputs blow this long before the step count looks
  unusual.
* **wall clock** — neither other ceiling moves while a command sits waiting on
  a timeout.

Checked **before** each step, never during. Interrupting a tool halfway leaves
the workspace in a state nobody planned for; letting it finish costs one extra
step and ends with a state the trace explains.

At 75% of the tightest ceiling the agent is told how many steps remain. An
agent that knows it has three steps left writes its answer; one that does not
keeps exploring and gets cut off mid-thought with nothing to show.

A run that ends on a ceiling reports `BUDGET_EXHAUSTED` and returns the last
useful observation, **labelled partial**. Reporting nothing throws away real
work; reporting it as a finished answer is worse.

---

## Context

**`context_char_budget`, `keep_recent_steps`, `compact`, `max_observation_chars`**

Context growth is what kills long runs. Each step re-sends everything before
it, so cost is quadratic in steps, and the largest thing in the transcript is
tool output. Three mechanisms, applied cheapest-first:

1. **Per-observation caps** (`max_observation_chars`) stop one result being
   enormous in the first place. Truncation is always announced in the result
   itself — silent truncation makes a model read half a file as the whole file
   and conclude the rest does not exist.
2. **Compaction** replaces the middle of the transcript with a digest once it
   outgrows `context_char_budget`, keeping the task pinned at the top and the
   last `keep_recent_steps` verbatim at the bottom.
3. **Hard trimming** is the backstop when even the kept tail is over budget.

The digest is **built by code, not by a model**: one line per call recording
what ran, with what arguments, and whether it worked. A summariser call costs
tokens and latency exactly when the run is already under pressure, and it can
invent a detail that then looks, to every later step, like something the agent
observed. The digest is lossy but cannot be wrong, and what an agent needs from
its distant past is "I already tried that", which it preserves exactly.

Recent steps are never compacted. The immediate past is what stops the agent
repeating the action that just failed; compacting it saves tokens and costs the
loop guard its reason for existing.

**Prompt caching.** The system prompt and tool definitions are byte-identical
on every call while the transcript grows underneath them — the textbook case
for a cache breakpoint. One is set at the end of the tool array, which covers
both. On a twenty-step run that is the difference between paying for the tool
schemas twenty times and paying once.

---

## Tools

**`enabled_tools`, `shell_allow`, `shell_timeout`**

`enabled_tools` is a real boundary, not a suggestion: a tool that is not built
cannot be called whatever the model asks for. Narrowing it to
`("read_file", "grep")` gives a genuinely read-only agent. The terminal tools
(`finish`, `give_up`) are always present — an agent with no way to say "done"
can only end by exhausting a budget, which turns every success into a reported
failure.

`run_command` takes the arguments but never the syntax: the command is parsed
with `shlex.split` and executed with `shell=False`, so `;`, `|`, `&&` and
`$(...)` are not operators. The program name must be on `shell_allow`. A
denylist would lose — `rm` blocked is `/bin/rm` allowed is `python3 -c "import
os; os.remove(...)"` allowed.

---

## Trace

**`trace`, `trace_dir`**

One JSONL event per line, flushed as it is written, never rewritten. A run that
crashes, hangs or is killed by a budget is precisely the run you want the trace
for, and a document assembled at the end is the one you will not have.

```bash
actcheck trace --list        # every recorded run
actcheck trace               # render the most recent
actcheck trace -v            # include tool output
actcheck replay              # re-execute it, free and deterministic
```

Next: [evaluation](04-evaluation.md).
