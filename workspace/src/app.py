"""Harbour order intake."""

from dataclasses import dataclass

VALID_STATUSES = ("pending", "shipped", "refunded", "cancelled")


@dataclass
class Order:
    order_id: str
    customer: str
    total_cents: int
    status: str


def validate(order: Order) -> list[str]:
    """Return a list of problems with the order; empty means it is valid."""
    problems = []
    if order.total_cents <= 0:
        problems.append("total must be positive")
    if order.status not in VALID_STATUSES:
        problems.append(f"unknown status: {order.status}")
    if not order.customer.strip():
        problems.append("customer is required")
    return problems


def total_for(orders: list[Order], status: str) -> int:
    """Sum the totals of every order in the given status."""
    return sum(o.total_cents for o in orders if o.status == status)
