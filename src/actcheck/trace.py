"""The run trace: append-only JSONL, one event per line.

If you take one thing from this project, take this file. An agent is a loop
whose behaviour you cannot reason about from its final answer, because the
answer is the one part that looks the same whether the run went well or went
sideways for nine steps and recovered. The trace is where the run is actually
visible.

Design choices and the reasons for them:

* **JSONL, appended as the run proceeds.** Not a JSON document written at the
  end. A run that crashes, hangs, or gets killed by a budget is precisely the
  run you most want the trace for, and a document assembled at the end is the
  one you will not have. Every line is flushed when it is written.
* **One event per line, each self-describing.** `jq`, `grep` and `tail -f` all
  work on it with no tooling of ours.
* **Append-only.** Nothing is ever rewritten -- a verdict arriving after a step
  is a new event referring back to it, not an edit. Mutable history is how you
  end up unable to tell what the agent knew at the time.

The format is stable enough to depend on: `read_trace` is what `actcheck
replay` and the eval harness both read.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from actcheck.types import Plan, RunResult, Step

TRACE_VERSION = 1


class Trace:
    """Writes one run's events to a JSONL file.

    Used as a context manager so the terminal event is written even when the
    body raises:

        with Trace(path, run_id, task) as trace:
            trace.step(step)
    """

    def __init__(
        self, path: str | Path, run_id: str, task: str, meta: dict[str, Any] | None = None
    ):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._started = time.time()
        self.write(
            "run_start",
            task=task,
            version=TRACE_VERSION,
            meta=meta or {},
        )

    # ----------------------------------------------------------------- write
    def write(self, event: str, **fields: Any) -> None:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "event": event, **fields}
        self._fh.write(json.dumps(record, default=str) + "\n")
        # Flushed per event on purpose: a trace you cannot read until the
        # process exits cleanly is a trace that is missing when it matters.
        self._fh.flush()

    def plan(self, plan: Plan, *, revision: int = 0) -> None:
        self.write(
            "plan",
            revision=revision,
            plan={
                "goal": plan.goal,
                "revisions": plan.revisions,
                "steps": [asdict(s) for s in plan.steps],
            },
        )

    def step(self, step: Step) -> None:
        self.write("step", **step.to_dict())

    def note(self, message: str, **fields: Any) -> None:
        """Something the loop did that is not a step: a guard firing, a
        compaction, a budget warning. These are what explain the gaps."""
        self.write("note", message=message, **fields)

    def finish(self, result: RunResult) -> None:
        self.write(
            "run_end",
            status=str(result.status),
            reason=result.reason,
            answer=result.answer,
            steps=result.n_steps,
            usage=asdict(result.usage),
            duration_ms=result.duration_ms,
        )

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> Trace:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None:
            self.write("error", error=f"{exc_type.__name__}: {exc}")
        self.close()


class NullTrace(Trace):
    """A trace that writes nothing, for `trace = false` and for tests.

    Subclassing rather than sprinkling `if self.trace:` through the agent: the
    loop should not have to know whether anyone is recording.
    """

    def __init__(self, *_: Any, **__: Any) -> None:  # noqa: D107 - see class docstring
        self.path = Path("/dev/null")
        self.run_id = ""

    def write(self, event: str, **fields: Any) -> None:
        return None

    def close(self) -> None:
        return None


# ------------------------------------------------------------------- reading
def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load a trace. Malformed lines are skipped, not fatal.

    A truncated final line is the normal state of a trace from a killed
    process, and refusing to read the other 200 events because of it would make
    the tool useless exactly when it is needed.
    """
    events: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def render_trace(events: list[dict[str, Any]], *, verbose: bool = False) -> str:
    """Human-readable rendering, as shown by `actcheck trace show`."""
    out: list[str] = []
    start = events[0]["ts"] if events else 0.0

    for event in events:
        kind = event.get("event")
        at = f"{event.get('ts', 0) - start:7.2f}s"

        if kind == "run_start":
            out.append(f"{at}  RUN   {event.get('task', '')}")
        elif kind == "plan":
            plan = event.get("plan", {})
            label = "PLAN" if not event.get("revision") else f"PLAN~{event['revision']}"
            out.append(f"{at}  {label}  {len(plan.get('steps', []))} steps")
            for s in plan.get("steps", []):
                out.append(f"{'':9}  {s['id']}. {s['description']}")
        elif kind == "step":
            out.append(f"{at}  STEP  {_render_step(event, verbose)}")
        elif kind == "note":
            out.append(f"{at}  NOTE  {event.get('message', '')}")
        elif kind == "error":
            out.append(f"{at}  ERROR {event.get('error', '')}")
        elif kind == "run_end":
            usage = event.get("usage", {})
            out.append(
                f"{at}  END   [{event.get('status')}] {event.get('reason', '')} "
                f"({event.get('steps')} steps, "
                f"{usage.get('input_tokens', 0) + usage.get('output_tokens', 0)} tokens)"
            )
    return "\n".join(out)


def _render_step(event: dict[str, Any], verbose: bool) -> str:
    call = event.get("tool_call") or {}
    obs = event.get("observation") or {}
    verdict = event.get("verdict") or {}

    name = call.get("name", "(no tool)")
    args = json.dumps(call.get("args", {}), default=str)
    if not verbose and len(args) > 70:
        args = args[:67] + "..."

    status = "ok" if obs.get("ok", True) else "ERR"
    mark = {"progress": "+", "step_done": "*", "goal_done": "#"}.get(
        verdict.get("judgement", ""), "-"
    )
    line = f"{event.get('index', '?'):>2} {mark} {name}{args}  [{status}]"
    if verdict.get("reason"):
        line += f"\n{'':15}{verdict['judgement']}: {verdict['reason']}"
    if verbose and obs.get("content"):
        body = "\n".join(f"{'':15}| {ln}" for ln in str(obs["content"]).splitlines()[:20])
        line += "\n" + body
    return line


def latest_trace(trace_dir: str | Path) -> Path | None:
    """Most recently written trace in `trace_dir`, or None."""
    directory = Path(trace_dir)
    if not directory.is_dir():
        return None
    traces = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return traces[0] if traces else None
