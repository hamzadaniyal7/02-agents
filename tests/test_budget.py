"""Budgets: the three ways a run is allowed to end early."""

from __future__ import annotations

import time

from actcheck.budget import Budget
from actcheck.types import Usage


def test_a_fresh_budget_is_not_exhausted() -> None:
    assert Budget(max_steps=3).exhausted() is None


def test_the_step_ceiling_names_itself() -> None:
    budget = Budget(max_steps=2)
    budget.record_step()
    budget.record_step()
    assert "step budget" in budget.exhausted()


def test_the_token_ceiling_names_itself() -> None:
    budget = Budget(max_tokens=100)
    budget.record_usage(Usage(input_tokens=80, output_tokens=40))
    assert "token budget" in budget.exhausted()


def test_the_time_ceiling_names_itself() -> None:
    budget = Budget(max_seconds=0.01)
    time.sleep(0.02)
    assert "time budget" in budget.exhausted()


def test_planner_and_checker_tokens_count_against_the_budget() -> None:
    """A budget that ignored them would understate the run by the cost of the
    checker, which is the single largest non-agent expense."""
    budget = Budget(max_tokens=100)
    budget.record_usage(Usage(input_tokens=60))  # planner
    budget.record_usage(Usage(input_tokens=60))  # checker
    assert budget.exhausted() is not None


def test_remaining_steps_never_goes_negative() -> None:
    budget = Budget(max_steps=1)
    budget.record_step()
    budget.record_step()
    assert budget.remaining_steps() == 0


def test_pressure_tracks_the_tightest_ceiling() -> None:
    budget = Budget(max_steps=4, max_tokens=1_000_000, max_seconds=1e6)
    budget.record_step()
    budget.record_step()
    budget.record_step()
    assert 0.74 < budget.pressure() <= 0.76


def test_pressure_is_capped_at_one() -> None:
    budget = Budget(max_steps=1)
    budget.record_step()
    budget.record_step()
    assert budget.pressure() == 1.0


def test_usage_totals_accumulate() -> None:
    usage = Usage()
    usage.add(Usage(input_tokens=10, output_tokens=5, calls=1))
    usage.add(Usage(input_tokens=1, output_tokens=2, calls=1))
    assert usage.total_tokens == 18 and usage.calls == 2


def test_snapshot_reports_every_ceiling() -> None:
    assert set(Budget().snapshot()) >= {"steps_used", "max_steps", "tokens_used", "pressure"}
