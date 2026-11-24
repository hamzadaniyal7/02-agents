"""Measuring the loop.

The question this answers is not "did the agent sound right?" but "did the run
end in the state the task required?", which is the only question that survives
contact with a non-deterministic model.

Every expectation here is **programmatic**: a string that must appear in the
answer, a file that must exist, a status the run must end in. No LLM judge is
involved in the pass/fail decision. A judge is a reasonable thing to add for
nuance -- and 01-rag has one for faithfulness -- but an agent task usually has
an observable outcome, and when it does, checking the outcome beats asking a
model for an opinion about it.

The metrics worth watching, and the failure each one catches:

* **success** -- ended in the expected status with the expected effects. The
  headline, and on its own not enough.
* **answer_match** -- the required strings are in the answer. Catches the run
  that did the work and then reported something else.
* **abstention** -- on the deliberately impossible tasks, did it call `give_up`
  instead of inventing an answer? An agent that scores well everywhere else and
  badly here is an agent that confabulates under pressure, which is worse than
  one that simply fails.
* **steps** -- how much it spent getting there. Two configurations with equal
  success and a 2x step gap are not equally good.
* **tool_error_rate** -- usually a prompt or tool-description problem, not a
  model problem.
* **stuck / budget_exhausted** -- how often the guard and the ceilings had to
  end the run. These should be rare and non-zero; zero usually means the tasks
  are too easy to be measuring anything.

Each task runs in a fresh copy of the workspace, so a task that writes files
cannot change the result of the next one.
"""

from __future__ import annotations

import json
import re
import shutil
import statistics
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from actcheck.agent import Agent
from actcheck.config import Config
from actcheck.types import RunResult, RunStatus

DEFAULT_TASKS = Path("eval/tasks.jsonl")


@dataclass(slots=True)
class TaskResult:
    """One task's outcome, with every expectation scored separately.

    Keeping the sub-scores rather than a single boolean is what makes a
    regression legible: "success dropped" is a mystery, "answer_match dropped
    while status stayed right" is a reporting bug.
    """

    task_id: str
    task: str
    passed: bool
    status: str
    expected_status: str
    answer_match: bool | None = None
    effects_match: bool | None = None
    steps: int = 0
    tokens: int = 0
    tool_errors: int = 0
    duration_ms: int = 0
    failures: list[str] = field(default_factory=list)
    answer: str = ""
    trace_path: str = ""


@dataclass(slots=True)
class EvalReport:
    results: list[TaskResult]
    config: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- metrics
    def metrics(self) -> dict[str, float | int]:
        n = len(self.results) or 1
        steps = [r.steps for r in self.results]
        answers = [r.answer_match for r in self.results if r.answer_match is not None]
        abstention = [r for r in self.results if r.expected_status in ("failed", "stuck")]

        return {
            "tasks": len(self.results),
            "success": round(sum(r.passed for r in self.results) / n, 4),
            "status_match": round(sum(r.status == r.expected_status for r in self.results) / n, 4),
            "answer_match": round(sum(answers) / len(answers), 4) if answers else 0.0,
            "abstention_correct": (
                round(sum(r.passed for r in abstention) / len(abstention), 4) if abstention else 0.0
            ),
            "steps_mean": round(statistics.mean(steps), 2) if steps else 0.0,
            "steps_max": max(steps) if steps else 0,
            "tool_error_rate": round(
                sum(r.tool_errors for r in self.results) / max(1, sum(steps)), 4
            ),
            "stuck": sum(r.status == "stuck" for r in self.results),
            "budget_exhausted": sum(r.status == "budget_exhausted" for r in self.results),
            "tokens_total": sum(r.tokens for r in self.results),
        }

    def render(self, *, verbose: bool = False) -> str:
        out = [f"tasks: {len(self.results)}", ""]
        width = max((len(k) for k in self.metrics()), default=10)
        for key, value in self.metrics().items():
            out.append(f"  {key:<{width}}  {value}")

        failures = [r for r in self.results if not r.passed]
        if failures:
            out.append("")
            out.append(f"failed {len(failures)}/{len(self.results)}:")
            for r in failures:
                out.append(f"  {r.task_id}: {'; '.join(r.failures) or r.status}")
                if verbose:
                    out.append(f"      task:   {r.task}")
                    out.append(f"      answer: {r.answer[:300]}")
        return "\n".join(out)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            {
                "metrics": self.metrics(),
                "config": self.config,
                "results": [asdict(r) for r in self.results],
            },
            indent=indent,
            default=str,
        )


# ----------------------------------------------------------------- running
def load_tasks(path: str | Path = DEFAULT_TASKS) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from None
    return tasks


def run_eval(
    cfg: Config,
    *,
    tasks_path: str | Path = DEFAULT_TASKS,
    only: str | None = None,
    on_task: Callable[[TaskResult], None] | None = None,
) -> EvalReport:
    """Run every task and score it."""
    tasks = load_tasks(tasks_path)
    if only:
        tasks = [t for t in tasks if only in t.get("id", "")]
    source = cfg.workspace_path

    results: list[TaskResult] = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix="actcheck-eval-") as tmp:
            # A fresh copy per task. Tasks that write files are the ones most
            # worth evaluating and the ones that would otherwise contaminate
            # every task after them.
            sandbox_dir = Path(tmp) / "workspace"
            shutil.copytree(source, sandbox_dir)
            task_cfg = cfg.merged({"workspace": str(sandbox_dir), **dict(task.get("config", {}))})
            result = Agent(task_cfg).run(task["task"])
            scored = _score(task, result, sandbox_dir)
        results.append(scored)
        if on_task is not None:
            on_task(scored)

    return EvalReport(results=results, config=cfg.to_dict())


def _score(task: dict[str, Any], result: RunResult, workspace: Path) -> TaskResult:
    expect = task.get("expect", {})
    expected_status = str(expect.get("status", RunStatus.SUCCESS))
    failures: list[str] = []

    status_ok = str(result.status) == expected_status
    if not status_ok:
        failures.append(f"status {result.status} != {expected_status}")

    answer_match: bool | None = None
    required = expect.get("answer_contains")
    if required:
        needles = [required] if isinstance(required, str) else list(required)
        haystack = result.answer.lower()
        missing = [n for n in needles if n.lower() not in haystack]
        answer_match = not missing
        if missing:
            failures.append(f"answer missing {missing}")

    # Substring checks are too generous for short answers: "3" is satisfied by
    # the order id "A-1003", which is how an eval ends up reporting a pass the
    # agent did not earn. Numeric expectations use a bounded pattern instead.
    patterns = expect.get("answer_regex")
    if patterns:
        patterns = [patterns] if isinstance(patterns, str) else list(patterns)
        unmatched = [p for p in patterns if not re.search(p, result.answer, re.IGNORECASE)]
        matched = not unmatched
        answer_match = matched if answer_match is None else (answer_match and matched)
        if unmatched:
            failures.append(f"answer does not match {unmatched}")

    effects_match: bool | None = None
    effects = expect.get("files", {})
    if effects:
        effects_match = True
        for rel, must_contain in effects.items():
            path = workspace / rel
            if not path.is_file():
                effects_match = False
                failures.append(f"expected file {rel} was not created")
                continue
            if (
                must_contain
                and must_contain.lower()
                not in path.read_text(encoding="utf-8", errors="replace").lower()
            ):
                effects_match = False
                failures.append(f"{rel} does not contain {must_contain!r}")

    max_steps = expect.get("max_steps")
    if max_steps is not None and result.n_steps > int(max_steps):
        failures.append(f"took {result.n_steps} steps, budgeted {max_steps}")

    return TaskResult(
        task_id=str(task.get("id", "?")),
        task=task["task"],
        passed=not failures,
        status=str(result.status),
        expected_status=expected_status,
        answer_match=answer_match,
        effects_match=effects_match,
        steps=result.n_steps,
        tokens=result.usage.total_tokens,
        tool_errors=result.tool_errors,
        duration_ms=result.duration_ms,
        failures=failures,
        answer=result.answer,
        trace_path=result.trace_path,
    )
