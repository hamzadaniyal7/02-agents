# Limits, and what comes next

What this does not do, stated plainly, because a roadmap that reads like a
feature list is marketing.

## Known limits

**One agent, one task.** There is no delegation, no sub-agents, no shared
state between runs. Multi-agent systems mostly multiply the failure modes in
[01](01-what-is-an-agent.md#why-the-naive-loop-fails) by the number of agents;
the single loop has to be trustworthy first.

**Tool calls run sequentially.** If a model returns several in one turn they
are executed in order. Parallel execution is real throughput for independent
reads, but it complicates the loop guard (which reasons about "the last call")
and the checker (which rules on one observation). Worth doing, not yet done.

**The checker is a single model call with no memory.** It sees the goal, the
step, the criterion and one observation. It cannot notice that the agent has
been circling a wrong assumption for five steps — that is the loop guard's job,
and the loop guard only sees repetition, not wrongness.

**Compaction is lossy in a way that can matter.** The digest keeps what was
tried and whether it worked, not what was *learned*. An agent that read a
config file forty steps ago and needs a value from it will have to read it
again. The deterministic digest is the right default
([ADR 4](05-decisions.md#adr-4-the-digest-is-built-by-code-not-by-a-model));
a fact-extraction pass over compacted turns would be strictly better and is not
implemented.

**No cost in currency.** Tokens are counted, money is not. Per-model pricing
changes often enough that a hardcoded table is a liability, and reading it from
config is a feature nobody asked for yet.

**The offline policy is not an agent.** It is a lexical matcher that exercises
the plumbing. `actcheck eval --provider scripted` scores 1/11, and
[04-evaluation.md](04-evaluation.md#the-offline-baseline) explains why that is
the honest number rather than a bug.

**The sandbox is not a security boundary.** See
[ADR 3](05-decisions.md#adr-3-the-sandbox-is-a-mistake-boundary-not-a-security-boundary).
Untrusted task descriptions need a container.

**Single-turn tasks only.** A run takes a task and ends. There is no way to
hand it a follow-up that reuses the workspace state and the transcript, which
is how anyone would actually use this interactively.

## What would come next, in order

**1. Parallel tool execution.** Independent reads are the common case and the
latency win is real. Needs the loop guard to reason about a set of calls rather
than the last one.

**2. A resume command.** `actcheck resume <trace> "now also do X"` — restore
the transcript from a trace and continue. The trace already contains
everything needed; the loop does not yet have an entry point that accepts it.

**3. Fact extraction during compaction.** A cheap pass that pulls stable facts
("the build command is `make build`") out of turns about to be compacted, and
pins them like the task statement. Keeps the epistemics of ADR 4 — extracted
quotes, not generated prose — while losing much less.

**4. A real-model baseline in the repo.** The numbers in the README are from
the offline provider because those are the ones anyone can reproduce without a
key. A recorded `--provider anthropic` run, committed as a trace and replayed
in CI, would give the regression signal that the offline eval cannot.

**5. Tool-level permissions per run.** `enabled_tools` is all-or-nothing per
tool. "May read anywhere, may write only under `out/`" is the shape people
actually want, and it belongs in the sandbox rather than in each tool.

**6. Streaming step output.** Steps print when they complete. A long
`run_command` shows nothing until it finishes, which is exactly when you most
want to see progress.

## What is deliberately not planned

**A web UI.** The trace is a file. `jq` and `tail -f` already work on it.

**A plugin system.** Adding a tool means writing a class with a `run()` and a
schema and putting it in `build_registry`. A plugin system would make that
harder to read in exchange for making it configurable by people who are not
reading the code.

**Multi-agent orchestration.** See the first limit above.

**More providers for their own sake.** `llm/base.py` is two methods; a second
provider is an afternoon. Adding one before anybody needs it buys an untested
code path.
