# 02-agents — an agent loop that knows when it is failing

Give it a task and it works through it one tool call at a time. The part worth
your attention is what happens when it *isn't* working:

```
$ actcheck run "Read nonexistent.md and report what it says."

   1 - read_file{"path": "nonexistent.md"}
       no_progress: tool failed: no such file or directory: nonexistent.md
   2 - read_file{"path": "nonexistent.md"}
       no_progress: tool failed: no such file or directory: nonexistent.md
   3 - read_file{"path": "nonexistent.md"}
       no_progress: You have now called read_file with identical arguments 3
       times in a row and the result has not changed. It will not change on
       another attempt. Either use a different tool or different arguments,
       or call give_up explaining what is blocking you.
   4 - read_file{"path": "nonexistent.md"}

[stuck] repeated the same read_file call after being told it would not help
  --  4 steps, 272 tokens, 2ms
```

The step budget was 24. It stopped at 4, said **why** in a sentence you can act
on, and the status is `stuck` — not `success`, and not the `budget_exhausted`
it would have reported after burning every remaining step to reach the same
conclusion.

Note the shape of that: the guard does not kill on first detection. It puts the
repetition **into the transcript** as the tool result, giving the agent one
turn to choose differently with the evidence in front of it. Only the identical
call *after* that ends the run.

That is the whole thesis. Agents that act are easy. The engineering is in the
loop around the acting: something that rules on whether a step helped,
something that notices going in circles, ceilings that stop a run early with
its reason named, and a record of every step as it happens.

Built from scratch — no LangChain, no agent SDK. About 4,500 lines of Python
across 26 small modules, a fifth of it comments and docstrings explaining *why*, with 244
tests that run offline in under two seconds.

---

## Quickstart

```bash
git clone https://github.com/hamzadaniyal7/02-agents && cd 02-agents
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -e '.[dev]'

actcheck tools                                        # what an agent gets
actcheck run "Which office does primary on-call sit in?" --provider scripted
```

That last command runs the complete loop — planning, tools, checking, guards,
budgets, trace — with **no API key and no network**. A sample workspace ships
with the repo for it to work in.

To use a real model:

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # or run `ant auth login`
actcheck run "Summarise what this service does and which files say so."
```

---

## What happens on each step

```
                   ┌──────────────────── transcript ◀───────────────┐
                   ▼                                                │
  budget check ──▶ model picks a tool                               │
       │                        │                                   │
       │ exhausted              ▼                                   │
       ▼                   loop guard ──── repeat? ──▶ nudge ───────┤
  BUDGET_EXHAUSTED              │ clear                             │
                                ▼                                   │
                        tool runs (sandboxed, timed, capped)        │
                                │                                   │
                                ▼                                   │
                     checker: did that help?                        │
                                │                                   │
              ┌─────────────────┼─────────────────┐                 │
              ▼                 ▼                 ▼                 │
          GOAL_DONE         advanced          stalled ──────────────┘
              │                                   │
           SUCCESS                      replan / nudge / STUCK
```

| Stage | What happens | Where |
|---|---|---|
| **Plan** | One call turns the task into ordered steps, each with a *checkable* success criterion. A step without one is a wish. | `planner.py` |
| **Act** | The model picks a tool; the registry runs it. Every observation is timed, size-capped and error-wrapped in one place. | `agent.py`, `tools/base.py` |
| **Check** | Cheap rules first (errored? empty? identical to last time?), then a separate cheaper model against the criterion. | `checker.py` |
| **Guard** | Identical calls, and stalls where every step is new and nothing moves. Nudges before it kills. | `loopguard.py` |
| **Budget** | Step, token and wall-clock ceilings, checked before each step. The stop reason names which one fired. | `budget.py` |
| **Context** | Observation caps, then compaction into a deterministic digest, then hard trimming. Recent steps are never compacted. | `context.py` |
| **Record** | Append-only JSONL, flushed per event. The run that crashes is the one you want this for. | `trace.py` |
| **Replan** | Only on a stall, handed the evidence, capped. Past the cap the run is `STUCK`. | `planner.py` |

Full write-ups in [`docs/`](docs/) — start with
[what an agent loop is](docs/01-what-is-an-agent.md), then
[the loop stage by stage](docs/03-the-loop.md).

---

## Every run leaves a trace

```
$ actcheck trace

   0.00s  RUN   Which office does primary on-call sit in?
   0.00s  PLAN  3 steps
           1. Locate the files relevant to the task
           2. Read the relevant content
           3. Report the answer
   0.00s  STEP   1 + grep{"pattern": "on-call", "ignore_case": true}  [ok]
               progress: offline checker: the tool returned content
   0.00s  STEP   2 + read_file{"path": "README.md"}  [ok]
               progress: offline checker: the tool returned content
   0.00s  STEP   3 + read_file{"path": "docs/runbook.md"}  [ok]
               progress: offline checker: the tool returned content
   0.00s  STEP   4 # finish{"answer": "Primary on-call sits with the Platform team in the Hels...  [ok]
               goal_done: agent reported the task complete
   0.00s  END   [success] agent called finish (4 steps, 2321 tokens)
```

*(That run used `--provider scripted`, so the verdicts are labelled "offline
checker". With a real model they carry its reasoning instead.)*

It is JSONL, one event per line, written as the run proceeds — so `jq`,
`grep` and `tail -f` all work on it, and a run killed mid-flight still has one.

And because every tool call is recorded, a run can be **re-executed exactly**:

```bash
actcheck replay            # same calls, same order, 0 tokens
```

That is the regression test an agent can actually have. Record a run that
worked, replay it in CI, and a change that breaks a tool's output format shows
up immediately — with no model in the loop to paper over the difference by
adapting to it.

---

## Commands

```bash
actcheck init                            # write a starter actcheck.toml
actcheck tools [--json]                  # what an agent would get, and its schemas
actcheck plan "task"                     # the plan, without touching the workspace
actcheck run "task" [--max-steps N] [--checker off] [--no-plan] [--json] [-q]
actcheck trace [path] [--list] [-v]      # inspect recorded runs
actcheck replay [path]                   # re-execute one, free and deterministic
actcheck eval [--only X] [--checker off] # score the loop against the task set
```

Two flags worth knowing:

* `--provider scripted` runs everything with no key and no cost. Use it to try
  the tool, and to check you have not broken the harness.
* `--checker off` turns off independent checking. It is the baseline the
  default configuration has to beat, and `actcheck eval --checker off` is how
  you find out whether it does on *your* tasks.

As a library:

```python
from actcheck import Agent, Config

result = Agent(Config.load()).run("Find the build command and report it.")

print(result.status)  # success | failed | stuck | budget_exhausted | aborted
print(result.answer)
print(result.n_steps, result.usage.total_tokens)
print(result.trace_path)

for step in result.steps:
    print(step.tool_call.name, "->", step.verdict.judgement, step.verdict.reason)
```

---

## Is it any good? Measure it

Tuning an agent by watching it work is how you end up with a loop nobody
trusts. The repo ships a task set ([`eval/tasks.jsonl`](eval/tasks.jsonl)) with
four kinds of case: single-file lookups, multi-hop questions spanning two
files, tasks scored by their **side effects on disk**, and deliberately
**impossible** ones where the only correct outcome is `give_up`.

Every expectation is programmatic — a pattern in the answer, a file that must
exist, a status the run must reach. No LLM judge decides pass or fail.

Run against the sample workspace with the offline stand-in model:

```
$ actcheck eval --provider scripted
tasks: 11

  tasks               11
  success             0.0909
  status_match        0.8182   <- the loop machinery works
  answer_match        0.125
  abstention_correct  0.0      <- never gives up; extracts the nearest line instead
  steps_mean          5.73
  tool_error_rate     0.0159
  stuck               0
  budget_exhausted    0
```

**One task out of eleven, and that number is the point.** The offline policy is
a lexical matcher: it picks the obviously-relevant tool and answers with the
line that best overlaps the task. It solves lookups where the answer is a
sentence sitting in a file. It cannot count, cannot sum, cannot join two files,
cannot write one, and cannot tell "this is absent" from "this is adjacent" —
hence `abstention_correct` at zero.

The useful part of that table is `status_match` at 0.82 against `success` at
0.09: **the plumbing is sound and the judgement is missing.** Runs start, steps
execute, verdicts land, traces get written, runs terminate correctly for the
right reason. That is exactly what an offline baseline should tell you, and it
is why CI gates on one task the policy genuinely solves rather than on the
whole set — weakening an eval until it passes is the one change that makes
every future number meaningless.

Those middle numbers are the gap a real model closes. Point the same harness at
`--provider anthropic` and the question becomes which configuration closes more
of it per token:

```bash
actcheck eval                        # the default: plan + both checkers
actcheck eval --checker programmatic # is the model checker earning its cost?
actcheck eval --no-plan              # is planning earning its cost?
```

Change one knob, run it again, see whether the number moved. That loop is the
difference between a demo and an engineered system.

---

## Configuration

Every knob lives in `actcheck.toml` (create one with `actcheck init`), and each
can be overridden by an `ACTCHECK_*` environment variable or a CLI flag.

```toml
[actcheck]
workspace = "workspace"     # tools cannot touch anything outside this

max_steps = 24              # three independent ceilings; whichever hits first
max_tokens = 200000         # ends the run, and the reason names it
max_seconds = 300
max_observation_chars = 6000

provider = "anthropic"      # anthropic | scripted | replay
model = "claude-opus-5"
effort = "medium"           # low | medium | high | xhigh | max

plan = true
max_replans = 2             # past this, a stalled run is STUCK, not re-planned again

checker = "both"            # programmatic | model | both | off
checker_model = "claude-sonnet-5"

repeat_threshold = 2        # identical calls in a row before the guard speaks up
no_progress_threshold = 4   # non-advancing steps before the run is called stuck

context_char_budget = 60000
keep_recent_steps = 6       # never compacted: it is what stops repeats
```

**On the sandbox.** Every path an agent supplies is fully resolved — symlinks
included — and checked for containment component by component, never by string
prefix (`"/work/../etc".startswith("/work")` is `True`, and so is
`/work-other`; both are tested). A workspace root that normalises to `/` or to
your home directory is refused outright. `run_command` parses with
`shlex.split` and runs with `shell=False`, so `;`, `|` and `$(...)` are not
operators, and the program name must be on an allowlist.

This is a boundary against expensive mistakes, **not** against a determined
adversary: an allowlisted `python3` can do anything Python can. Untrusted task
descriptions belong in a container. See
[ADR 3](docs/05-decisions.md#adr-3-the-sandbox-is-a-mistake-boundary-not-a-security-boundary).

**On read-only agents.** `enabled_tools = ["read_file", "grep"]` is a real
boundary, not a prompt instruction — a tool that is not built cannot be called
whatever the model asks for.

---

## Development

```bash
pytest                 # 244 tests, no API key required, ~1.5s
ruff check . && ruff format --check .
```

The suite runs entirely offline. A deterministic scripted model drives the real
loop, so the tests assert real behaviour — "a repeated call gets nudged instead
of re-executed", "a stuck run is not reported as budget-exhausted", "a symlink
pointing outside the workspace is rejected", "an entry count of 10 is not
mistaken for an empty directory" — with no network and no mocking of the parts
under test. CI runs the same thing on every push across Python 3.11, 3.12 and
3.13.

---

## Reading order

| Doc | What it covers |
|---|---|
| [docs/01-what-is-an-agent.md](docs/01-what-is-an-agent.md) | What a loop is, when you want one, and the four ways the naive version fails |
| [docs/02-architecture.md](docs/02-architecture.md) | Module map, data model, where each decision lives |
| [docs/03-the-loop.md](docs/03-the-loop.md) | Every stage, every knob, and the failure mode it addresses |
| [docs/04-evaluation.md](docs/04-evaluation.md) | What is measured, what each metric catches, how to tune with it |
| [docs/05-decisions.md](docs/05-decisions.md) | ADRs — the trade-offs, including the ones rejected |
| [docs/06-roadmap.md](docs/06-roadmap.md) | Known limits and what comes next |

## Where this sits

Project 2 of 3 in a build-it-yourself AI-engineering series:

| | Project | What it teaches |
|---|---|---|
| 01 | [rag](https://github.com/hamzadaniyal7/01-rag) | Retrieval, grounding, and citations you can verify |
| **02** | **agents** *(this repo)* | Loop engineering — plan → act → check over many steps |
| 03 | deploy | Serving an open model yourself with vLLM or SGLang |

## License

MIT — see [LICENSE](LICENSE).
