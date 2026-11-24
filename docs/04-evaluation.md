# Evaluation

Tuning an agent by watching it work is how you end up with a loop nobody
trusts. Every claim in this repo's README is a number from `actcheck eval`, and
this is how it is produced.

## The question

Not "did the agent sound right?" but **"did the run end in the state the task
required?"** — the only question that survives contact with a non-deterministic
model.

Every expectation is programmatic: a string in the answer, a pattern it must
match, a file that must exist with certain contents, a status the run must end
in. No LLM judge decides pass or fail. A judge is a reasonable thing to add for
nuance — [01-rag](https://github.com/hamzadaniyal7/01-rag) has one for
faithfulness — but an agent task usually has an observable outcome, and when it
does, checking the outcome beats asking a model for an opinion about it.

## The task set

`eval/tasks.jsonl`, one JSON object per line:

```json
{"id": "refunded-count",
 "task": "How many orders in data/orders.csv have the status 'refunded'?",
 "expect": {"status": "success", "answer_regex": ["\\b3\\b"], "max_steps": 8}}
```

| Field | Meaning |
|---|---|
| `status` | terminal status the run must reach |
| `answer_contains` | strings that must appear (case-insensitive) |
| `answer_regex` | patterns the answer must match |
| `files` | `{path: required substring}` checked on disk afterwards |
| `max_steps` | step ceiling; exceeding it fails the task |
| `config` | per-task config overrides |

Four kinds of task ship with the repo, and the mix is the point:

* **single-file lookups** — can it find and report one fact?
* **multi-hop** — "release 2.3.0 added a status; how many orders have it?"
  requires joining two files, and fails if the agent stops at the first.
* **side effects** — "create `summary.md` containing…"; scored on disk, not on
  what the agent claims.
* **impossible** — facts the workspace does not contain. The only correct
  outcome is `give_up`.

That last group is the one worth adding to your own set first. An agent that
scores well everywhere else and badly there is an agent that confabulates under
pressure, which is worse than one that simply fails.

### Why `answer_regex` exists

`answer_contains: ["3"]` is satisfied by the order id `A-1003`. The first
version of this eval reported two passes on that basis. Short answers —
numbers especially — need `answer_regex: ["\\b3\\b"]`, and a scorer that
flatters itself is worse than no scorer at all. There is a test asserting
exactly this (`test_a_substring_check_would_flatter_a_short_answer`).

## The metrics

```bash
actcheck eval                       # the full set
actcheck eval --provider scripted   # offline, free, deterministic
actcheck eval --only absent         # just the impossible ones
actcheck eval --checker off         # measure what checking buys you
actcheck eval --no-plan             # measure what planning buys you
```

| Metric | What it catches |
|---|---|
| `success` | the headline: right status, right answer, right effects, inside budget |
| `status_match` | ended the right *way* — separates "wrong answer" from "wrong ending" |
| `answer_match` | did the work, reported something else |
| `abstention_correct` | invents answers when the information is absent |
| `steps_mean` / `steps_max` | two configs with equal success and a 2x step gap are not equally good |
| `tool_error_rate` | usually a prompt or tool-description problem, not a model problem |
| `stuck` | how often the guard had to end a run |
| `budget_exhausted` | how often a ceiling did |

`stuck` and `budget_exhausted` should be **rare and non-zero**. Zero across a
whole set usually means the tasks are too easy to be measuring anything.

## The offline baseline

The scripted provider runs the entire loop with no API key. On the shipped task
set:

```
$ actcheck eval --provider scripted
tasks: 11

  tasks               11
  success             0.0909
  status_match        0.8182
  answer_match        0.125
  abstention_correct  0.0      <- never gives up; extracts the nearest line instead
  steps_mean          5.73
  tool_error_rate     0.0159
  stuck               0
  budget_exhausted    0
```

One task out of eleven. That number is the point.

The offline policy is a lexical matcher: it picks the obviously-relevant tool,
reads what it finds, and answers with the line that best overlaps the task. It
solves single-file lookups where the answer is a sentence sitting in a file.
It cannot count (`refunded-count`), cannot sum (`refunded-total`), cannot join
two files (`status-added-230`), cannot write one (`write-summary`), and cannot
tell "this is absent" from "this is adjacent" (`absent-*`, abstention 0.0).

`status_match` at 0.82 against `success` at 0.09 is the useful part of that
table: **the loop machinery works — runs start, steps execute, verdicts land,
traces are written, runs terminate correctly — and the model judgement is
missing.** That is exactly the split you want from an offline baseline, and it
is why CI gates on one task the policy genuinely solves rather than on the
whole set. Weakening an eval until it passes is the one change that makes every
future number meaningless.

Those middle numbers are the gap a real model closes. Point the same harness at
`--provider anthropic` and the interesting question becomes which
configuration closes more of it per token.

## The loop you actually want

```bash
actcheck eval --provider scripted --no-save   # fast, free: did I break the harness?
actcheck eval --checker programmatic          # cheaper: is the model checker earning its cost?
actcheck eval                                 # the real number
```

Change one knob, run it again, see whether the number moved. That loop is the
difference between a demo and an engineered system.

## Reading a failure

`eval/reports/*.json` keeps every run's sub-scores and its trace path:

```bash
actcheck trace .actcheck/traces/run_93c1c03af516.jsonl -v
```

Most "the agent is wrong" bugs are visible in the first three steps — a grep
that found nothing, a file read that was the wrong file. The trace is where you
see that; the answer is not.

Next: [the decisions and their trade-offs](05-decisions.md).
