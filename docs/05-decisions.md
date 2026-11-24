# Decisions

Architecture decision records: what was chosen, what it cost, and what was
rejected. The rejected options are the useful part.

---

## ADR 1: The checker is not the agent

**Decision.** Progress is judged by something other than the model doing the
work — programmatic rules first, then a separate, cheaper model.

**Why.** Ask a model whether it completed the task and it says yes. It is the
same model that chose the failing approach, reading the same transcript, with
no independent evidence. Self-assessment is not a check; it is the same opinion
asked twice.

**Rejected: the agent reports its own progress.** Free, and it is what most
loops do. It also means a run that went nowhere produces a confident summary of
nine steps of nothing, which is the single most damaging agent failure because
it is invisible downstream.

**Rejected: a model check on every step.** Roughly doubles the cost to learn
things the rules already knew with certainty — the tool errored, the result was
empty, the bytes are identical to last time. Most real failures are that
boring.

**Cost.** One extra model call on steps where the rules are inconclusive, and a
second place where a prompt can be wrong. `checker = "off"` is the baseline to
beat; `actcheck eval --checker off` measures what it buys you on your tasks.

---

## ADR 2: Guard first, kill second

**Decision.** On detecting repetition or a stall, put a message in the
transcript naming what just happened. Only end the run if that fails.

**Why.** The model is not broken when it repeats itself — it is doing the most
plausible next thing given a transcript in which that thing has not yet worked.
Telling it what it did converts a hidden pattern into visible evidence, which
is usually enough. Killing on first detection throws away a run that was one
sentence from recovering.

**Rejected: kill immediately.** Simpler, and wrong often enough to matter.

**Rejected: never kill, let budgets handle it.** Burns the entire remaining
budget to arrive at the same place, and reports `BUDGET_EXHAUSTED` when the
honest status is `STUCK`. Those are different problems and should not share a
status.

**Cost.** One wasted step per nudge, and a guard that can be nudged into
silence by a model that varies its arguments slightly. The no-progress detector
exists to catch exactly that case.

---

## ADR 3: The sandbox is a mistake boundary, not a security boundary

**Decision.** Confine tools to a workspace root, allowlist shell programs, and
say plainly that this stops accidents rather than adversaries.

**Why.** The confinement itself is real: paths are fully resolved (symlinks
included) and checked for containment component by component, not with a string
prefix — `"/work/../etc".startswith("/work")` is `True`, and so is
`/work-other`. Both are tested. A root that normalises to `/` or to your home
directory is refused outright, because confinement that confines an agent to
the whole machine is not confinement.

But an allowlisted `python3` can do anything Python can do, and `git` reaches
the network. Claiming more than that would be worse than claiming nothing,
because someone would rely on it.

**Rejected: a denylist of dangerous commands.** Denylists lose. `rm` blocked is
`/bin/rm` allowed is `python3 -c "import os; os.remove(...)"` allowed.

**Rejected: no shell at all.** Defensible, and it removes most of the value —
running the test suite is the single most useful thing an agent can do in a
code workspace.

**Cost.** Running untrusted task descriptions needs a container. This is not
that, and `tools/shell.py` says so where someone reading the code will see it.

---

## ADR 4: The digest is built by code, not by a model

**Decision.** When the transcript is compacted, older steps are replaced by a
deterministic line-per-call summary rather than a model-written one.

**Why.** A summariser call costs tokens and latency at exactly the moment the
run is already under budget pressure. Worse, it can invent a detail that then
looks, to every subsequent step, like something the agent actually observed —
a hallucination laundered into the evidence record. The deterministic digest is
lossy but cannot be wrong, and what an agent needs from its distant past is "I
already tried that, and it failed", which it preserves exactly.

**Rejected: a model summariser.** Better prose, strictly worse epistemics. The
hook is `Transcript(summarise=...)` if you want to measure the trade rather
than take my word for it.

**Rejected: a sliding window that drops old turns.** Cheapest, and it silently
loses the record of what was already tried — which is the one thing that stops
the agent going back round the same loop.

**Cost.** The digest reads like a log, not a narrative.

---

## ADR 5: The transcript uses the provider's message shape

**Decision.** Carry the conversation in Anthropic's content-block format
(`{"role", "content": [{"type": "tool_use", ...}]}`) rather than a neutral
internal representation.

**Why.** It is the most expressive of the shapes being targeted, and a lossy
intermediate that has to be translated back adds a layer that can only lose
information. Concretely: thinking blocks carry a signature the API validates on
the following turn, so an assistant turn rebuilt from parsed text is a rejected
request. Passing blocks through verbatim is not an optimisation, it is a
correctness requirement.

**Rejected: a neutral `Message` dataclass.** Cleaner on paper. In practice it
means a translation layer on both sides and a bug class where something
provider-specific is dropped in the round trip.

**Cost.** A provider for a different API has to translate. `llm/scripted.py`
shows how little that involves.

---

## ADR 6: Failures are values, not exceptions

**Decision.** Tools raise `ToolError`; the registry converts it into a failed
`Observation` that goes into the transcript in the same shape as a success.

**Why.** A tool that failed is information the model needs in order to choose
differently. The error message is written *for the model* — it says what was
wrong and usually what would work instead. An exception that unwinds the loop
denies it that chance and turns a recoverable step into a dead run.

Unexpected exceptions (a bug in a tool, not a decision by the model) are caught
too, reported with their type name, and the run continues. A tool bug should
not cost you the other nineteen steps.

**Rejected: let exceptions propagate.** One malformed path argument ends the
run.

**Rejected: retry with backoff.** The mechanism that turns one bad decision
into five identical ones. There is no retry anywhere in this loop.

**Cost.** Every tool author has to write error messages worth reading. The
base class makes that the obvious thing to do; nothing enforces it.

---

## ADR 7: Three budgets, not one

**Decision.** Independent step, token and wall-clock ceilings, checked before
each step, with the stop reason naming which one fired.

**Why.** They fail for different reasons. A loop making cheap calls and getting
nowhere never approaches a token ceiling. Long tool outputs blow a token
ceiling while the step count looks normal. Neither moves while a command sits
waiting on a timeout.

Checked **before** a step, never during: interrupting a tool halfway leaves the
workspace in a state nobody planned for.

**Rejected: a single cost ceiling.** Simpler to explain, and it collapses three
distinguishable failures into one undiagnosable one.

**Cost.** Three numbers to tune instead of one, and a run can overshoot a
ceiling by one step.

---

## ADR 8: The trace is written as the run happens

**Decision.** Append-only JSONL, flushed per event, never rewritten.

**Why.** The run you most want a trace for is the one that crashed, hung, or
got killed. A document assembled at the end is the one you will not have. Line
orientation means `jq`, `grep` and `tail -f` all work on it with no tooling of
ours. Append-only means a verdict arriving after a step is a new event
referring back to it, never an edit — mutable history leaves you unable to tell
what the agent knew at the time.

**Rejected: a JSON document written at the end.** Prettier, and absent exactly
when it matters.

**Cost.** One flush per event, and readers must tolerate a truncated final line
(`read_trace` does, and it is tested).

---

## ADR 9: No framework

**Decision.** No LangChain, no LangGraph, no agent SDK. About 4,500 lines of
Python across 26 modules, roughly a fifth of it comments and docstrings
explaining why.

**Why.** The mechanisms in this repo — loop guard, checker, compaction policy,
budget enforcement — are exactly what a framework abstracts away. The point is
to see them. "Agents are easy, reliable agents are hard" is only a useful thing
to say if the hard parts are legible.

**Rejected: build on an existing framework.** Faster to a working demo, and it
makes the interesting code someone else's.

**Cost.** No ecosystem, no integrations, and the tool set is deliberately
small.

Next: [known limits and what comes next](06-roadmap.md).
