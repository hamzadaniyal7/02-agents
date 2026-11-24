"""The command line.

Two principles, both learned the hard way from watching people use agent tools:

* **Show the steps as they happen.** A loop that prints nothing for ninety
  seconds and then produces an answer gives you no way to tell "thinking" from
  "stuck in a retry", and no way to kill it early when you can already see it
  has gone the wrong way. Every step prints a line as it completes.
* **Make the dry run free.** `actcheck plan` shows what the agent intends to do
  before any tool touches the workspace, and `--provider scripted` runs the
  entire loop with no key and no cost. Being able to try the thing is what gets
  it tried.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from actcheck.agent import Agent
from actcheck.config import CONFIG_FILENAME, Config
from actcheck.errors import ActcheckError
from actcheck.evaluate import DEFAULT_TASKS, TaskResult, run_eval
from actcheck.llm import build_model
from actcheck.planner import Planner
from actcheck.tools import build_registry
from actcheck.trace import latest_trace, read_trace, render_trace
from actcheck.types import Step

STARTER_CONFIG = """\
# actcheck configuration -- see docs/03-the-loop.md for what each knob does.
[actcheck]
workspace = "workspace"

# Budgets: the run stops at whichever ceiling it hits first.
max_steps = 24
max_tokens = 200000
max_seconds = 300

# Model
provider = "anthropic"      # anthropic | scripted | replay
model = "claude-opus-5"
effort = "medium"           # low | medium | high | xhigh | max

# Planning
plan = true
plan_max_steps = 8
max_replans = 2

# Checking: programmatic | model | both | off
checker = "both"
checker_model = "claude-sonnet-5"

# Loop guard
repeat_threshold = 2
no_progress_threshold = 4

# Context
context_char_budget = 60000
keep_recent_steps = 6
"""


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except ActcheckError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


# --------------------------------------------------------------------- parser
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="actcheck",
        description="An agent loop that plans, acts, checks its work, and records every step.",
    )
    sub = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help=f"path to {CONFIG_FILENAME}")
    common.add_argument("--workspace", help="sandbox root the tools are confined to")
    common.add_argument("--provider", choices=["anthropic", "scripted"], help="model provider")
    common.add_argument("--model", help="model id")
    common.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])

    run = sub.add_parser("run", parents=[common], help="run a task to completion")
    run.add_argument("task", help="what the agent should do")
    run.add_argument("--max-steps", type=int, help="step ceiling for this run")
    run.add_argument("--max-seconds", type=float, help="wall-clock ceiling for this run")
    run.add_argument("--checker", choices=["programmatic", "model", "both", "off"])
    run.add_argument("--no-plan", action="store_true", help="skip planning")
    run.add_argument("--no-trace", action="store_true", help="do not write a trace file")
    run.add_argument("--json", action="store_true", help="emit the full result as JSON")
    run.add_argument("--quiet", "-q", action="store_true", help="only print the final answer")
    run.set_defaults(func=cmd_run)

    plan = sub.add_parser("plan", parents=[common], help="show the plan without running it")
    plan.add_argument("task")
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(func=cmd_plan)

    tools = sub.add_parser("tools", parents=[common], help="list the tools an agent would get")
    tools.add_argument("--json", action="store_true")
    tools.set_defaults(func=cmd_tools)

    trace = sub.add_parser("trace", parents=[common], help="inspect recorded runs")
    trace.add_argument("path", nargs="?", help="trace file; defaults to the most recent")
    trace.add_argument("--list", action="store_true", help="list recorded traces")
    trace.add_argument("--verbose", "-v", action="store_true", help="include tool output")
    trace.set_defaults(func=cmd_trace)

    replay = sub.add_parser("replay", parents=[common], help="re-execute a recorded run")
    replay.add_argument("path", nargs="?", help="trace file; defaults to the most recent")
    replay.add_argument("--quiet", "-q", action="store_true")
    replay.set_defaults(func=cmd_replay)

    ev = sub.add_parser("eval", parents=[common], help="score the loop against the task set")
    ev.add_argument("--tasks", default=str(DEFAULT_TASKS))
    ev.add_argument("--only", help="run only tasks whose id contains this")
    ev.add_argument("--checker", choices=["programmatic", "model", "both", "off"])
    ev.add_argument("--no-plan", action="store_true")
    ev.add_argument("--no-save", action="store_true", help="do not write a report file")
    ev.add_argument("--json", action="store_true")
    ev.add_argument("--verbose", "-v", action="store_true")
    ev.set_defaults(func=cmd_eval)

    init = sub.add_parser("init", help=f"write a starter {CONFIG_FILENAME}")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=cmd_init)

    return parser


def _config(args: argparse.Namespace, **extra: object) -> Config:
    overrides = {
        "workspace": getattr(args, "workspace", None),
        "provider": getattr(args, "provider", None),
        "model": getattr(args, "model", None),
        "effort": getattr(args, "effort", None),
        "checker": getattr(args, "checker", None),
        "max_steps": getattr(args, "max_steps", None),
        "max_seconds": getattr(args, "max_seconds", None),
        **extra,
    }
    if getattr(args, "no_plan", False):
        overrides["plan"] = False
    if getattr(args, "no_trace", False):
        overrides["trace"] = False
    return Config.load(getattr(args, "config", None), **overrides)


# ------------------------------------------------------------------ commands
def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    hook = None if args.quiet or args.json else _print_step
    result = Agent(cfg, on_step=hook).run(args.task)

    if args.json:
        print(result.to_json())
    else:
        if not args.quiet:
            print()
        print(result.render() if not args.quiet else result.answer)
    # A non-zero exit on a failed run is what makes actcheck usable in a script
    # or a CI step without parsing its output.
    return 0 if result.ok else 1


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = _config(args)
    registry = build_registry(cfg)
    plan = Planner(build_model(cfg), cfg).plan(args.task, registry)
    if args.json:
        print(json.dumps({"goal": plan.goal, "steps": [vars(s) for s in plan.steps]}, default=str))
    else:
        print(f"goal: {plan.goal}\n")
        print(plan.render())
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    cfg = _config(args)
    registry = build_registry(cfg)
    if args.json:
        print(json.dumps(registry.specs(), indent=2))
        return 0
    print(f"workspace: {cfg.workspace_path}\n")
    for tool in registry:
        flag = "  (mutates)" if tool.mutates else ""
        params = ", ".join(tool.schema.get("properties", {}))
        print(f"  {tool.name}({params}){flag}")
        print(f"      {tool.description.strip().splitlines()[0]}")
    return 0


def cmd_trace(args: argparse.Namespace) -> int:
    cfg = _config(args)
    if args.list:
        directory = Path(cfg.trace_dir)
        traces = sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not traces:
            print(f"no traces in {directory}")
            return 0
        for path in traces:
            events = read_trace(path)
            task = next((e.get("task", "") for e in events if e.get("event") == "run_start"), "")
            end = next((e for e in events if e.get("event") == "run_end"), {})
            print(f"  {path.name}  [{end.get('status', 'incomplete')}]  {task[:60]}")
        return 0

    path = Path(args.path) if args.path else latest_trace(cfg.trace_dir)
    if not path or not path.is_file():
        print(f"no trace found (looked in {cfg.trace_dir})", file=sys.stderr)
        return 2
    print(render_trace(read_trace(path), verbose=args.verbose))
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    cfg = _config(args)
    path = Path(args.path) if args.path else latest_trace(cfg.trace_dir)
    if not path or not path.is_file():
        print(f"no trace found (looked in {cfg.trace_dir})", file=sys.stderr)
        return 2

    events = read_trace(path)
    task = next((e.get("task", "") for e in events if e.get("event") == "run_start"), "")
    if not task:
        print(f"{path} has no run_start event", file=sys.stderr)
        return 2

    cfg = cfg.merged({"provider": "replay"})
    print(f"replaying {path.name}: {task}\n")
    result = Agent(cfg, replay_path=str(path), on_step=None if args.quiet else _print_step).run(
        task
    )
    print()
    print(result.render())
    return 0 if result.ok else 1


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = _config(args)
    hook = None if args.json else _print_task
    report = run_eval(cfg, tasks_path=args.tasks, only=args.only, on_task=hook)

    if args.json:
        print(report.to_json())
    else:
        print()
        print(report.render(verbose=args.verbose))

    if not args.no_save:
        out = Path("eval/reports")
        out.mkdir(parents=True, exist_ok=True)
        name = f"{cfg.provider}-{cfg.checker}{'-noplan' if not cfg.plan else ''}.json"
        (out / name).write_text(report.to_json(), encoding="utf-8")
        if not args.json:
            print(f"\nreport: eval/reports/{name}")

    return 0 if report.metrics()["success"] == 1.0 else 1


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(CONFIG_FILENAME)
    if path.exists() and not args.force:
        print(f"{path} already exists (use --force to overwrite)", file=sys.stderr)
        return 2
    path.write_text(STARTER_CONFIG, encoding="utf-8")
    print(f"wrote {path}")
    return 0


# ------------------------------------------------------------------ printing
def _print_step(step: Step) -> None:
    """One line per step, to stderr so `--json` on stdout stays machine-readable."""
    call = step.tool_call
    verdict = step.verdict
    mark = {"progress": "+", "step_done": "*", "goal_done": "#", "blocked": "!"}.get(
        str(verdict.judgement) if verdict else "", "-"
    )
    args = json.dumps(call.args, default=str) if call else "{}"
    if len(args) > 60:
        args = args[:57] + "..."
    line = f"  {step.index:>2} {mark} {call.name if call else '?'}{args}"
    if verdict and not verdict.advanced:
        line += f"\n       {verdict.judgement}: {verdict.reason[:100]}"
    print(line, file=sys.stderr, flush=True)


def _print_task(result: TaskResult) -> None:
    mark = "pass" if result.passed else "FAIL"
    print(
        f"  [{mark}] {result.task_id:<22} {result.steps:>2} steps  {result.status}",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
