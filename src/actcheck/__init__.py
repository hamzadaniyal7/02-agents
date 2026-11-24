"""actcheck -- an agent loop that plans, acts, checks its own work, and records
every step in a replayable trace.

    from actcheck import Agent, Config

    agent = Agent(Config.load())
    result = agent.run("Summarise what this project does, citing the files you read.")
    print(result.answer)
    print(result.status, result.n_steps, result.trace_path)
"""

from __future__ import annotations

from actcheck.agent import Agent
from actcheck.budget import Budget
from actcheck.checker import Checker
from actcheck.config import Config
from actcheck.errors import ActcheckError, BudgetError, ConfigError, SandboxError, ToolError
from actcheck.loopguard import LoopGuard
from actcheck.planner import Planner
from actcheck.tools import Registry, Sandbox, build_registry
from actcheck.trace import Trace, read_trace, render_trace
from actcheck.types import (
    Judgement,
    Observation,
    Plan,
    PlanStep,
    RunResult,
    RunStatus,
    Step,
    StepStatus,
    ToolCall,
    Usage,
    Verdict,
)

__version__ = "0.1.0"

__all__ = [
    "ActcheckError",
    "Agent",
    "Budget",
    "BudgetError",
    "Checker",
    "Config",
    "ConfigError",
    "Judgement",
    "LoopGuard",
    "Observation",
    "Plan",
    "PlanStep",
    "Planner",
    "Registry",
    "RunResult",
    "RunStatus",
    "Sandbox",
    "SandboxError",
    "Step",
    "StepStatus",
    "ToolCall",
    "ToolError",
    "Trace",
    "Usage",
    "Verdict",
    "__version__",
    "build_registry",
    "read_trace",
    "render_trace",
]
