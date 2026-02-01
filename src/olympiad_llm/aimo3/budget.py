from __future__ import annotations

"""Time budget helpers for AIMO3.

The common failure mode in Kaggle is spending the entire per-problem budget on
exploration attempts, leaving too little time for second-stage verification.

We address this by reserving a small slice of time for verification.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TimeBudgetTracker:
    """Track actual time usage and dynamically adjust per-problem budgets.
    
    Key insight: if early problems solve quickly, we bank that time for harder problems.
    """
    
    total_budget_s: float
    total_problems: int
    base_timeout_s: float = 300.0
    high_timeout_s: float = 900.0
    # How much of banked time to use per problem (conservative = last longer)
    bank_utilization: float = 0.4
    # Minimum budget per problem (never go below this)
    min_budget_s: float = 120.0
    
    # State tracking
    problems_solved: int = field(default=0, init=False)
    total_time_used_s: float = field(default=0.0, init=False)
    solve_times: List[float] = field(default_factory=list, init=False)
    
    @property
    def problems_remaining(self) -> int:
        return max(0, self.total_problems - self.problems_solved)
    
    @property
    def time_remaining_s(self) -> float:
        return max(0.0, self.total_budget_s - self.total_time_used_s)
    
    @property
    def expected_time_used_s(self) -> float:
        """Time we expected to use based on base_timeout per problem."""
        return self.problems_solved * self.base_timeout_s
    
    @property
    def time_banked_s(self) -> float:
        """Time saved compared to expectation (can be negative if over budget)."""
        return self.expected_time_used_s - self.total_time_used_s
    
    @property
    def avg_solve_time_s(self) -> float:
        """Rolling average of actual solve times."""
        if not self.solve_times:
            return self.base_timeout_s
        return sum(self.solve_times) / len(self.solve_times)
    
    def compute_budget(self) -> float:
        """Compute the budget for the next problem based on current state."""
        if self.problems_remaining <= 0:
            return 0.0
        
        # Base: equal division of remaining time
        equal_share = self.time_remaining_s / self.problems_remaining
        
        # If we have banked time, we can be more generous
        if self.time_banked_s > 0:
            # Distribute banked time across remaining problems, with utilization factor
            bank_bonus = (self.time_banked_s * self.bank_utilization) / self.problems_remaining
            budget = equal_share + bank_bonus
        else:
            # We're behind schedule - stick to equal share (or less)
            budget = equal_share
        
        # Apply bounds
        budget = max(self.min_budget_s, budget)
        budget = min(self.high_timeout_s, budget)
        
        # Never exceed remaining time
        budget = min(budget, self.time_remaining_s)
        
        return budget
    
    def record_solve(self, time_used_s: float) -> None:
        """Record completion of a problem."""
        self.problems_solved += 1
        self.total_time_used_s += time_used_s
        self.solve_times.append(time_used_s)
        # Keep rolling window of last 10 for average
        if len(self.solve_times) > 10:
            self.solve_times.pop(0)
    
    def status_summary(self) -> str:
        """Return a short status string for logging."""
        return (
            f"[Budget] {self.problems_solved}/{self.total_problems} done | "
            f"Bank: {self.time_banked_s:+.0f}s | "
            f"Avg: {self.avg_solve_time_s:.0f}s | "
            f"Next: {self.compute_budget():.0f}s"
        )


def reserve_fraction_for_budget(
    *,
    budget_s: float,
    base_fraction: float,
    min_fraction: float = 0.10,
    max_fraction: float = 0.30,
) -> float:
    """Return an adaptive reserve fraction based on the total per-problem budget.

    Intuition (general):
    - for short budgets, keep relatively more reserve so verification still happens
    - for long budgets, reserve can be smaller because you have room for both generation and verification
    """

    b = max(0.0, float(budget_s))
    base = max(0.0, min(0.95, float(base_fraction)))
    lo = max(0.0, min(0.95, float(min_fraction)))
    hi = max(lo, min(0.95, float(max_fraction)))

    if b <= 120.0:
        # Very tight: keep a healthy reserve.
        return max(base, hi)
    if b >= 600.0:
        # Plenty of room: don't over-reserve.
        return min(base, lo)
    return base


def adaptive_verify_budget(
    *,
    remaining_s: float,
    base_fraction: float,
    cap_s: float,
    multiplier: float,
    min_s: float = 0.0,
) -> float:
    """Compute a second-stage verification budget from remaining time.

    This is adaptive via the multiplier (uncertainty-aware), but remains fully general.
    """

    remaining = max(0.0, float(remaining_s))
    if remaining <= 0:
        return 0.0

    frac = max(0.0, min(1.0, float(base_fraction)))
    cap = max(0.0, float(cap_s))
    mult = max(0.1, min(3.0, float(multiplier)))
    min_budget = max(0.0, float(min_s))

    budget = remaining * frac * mult
    if cap > 0:
        budget = min(budget, cap)
    if min_budget > 0:
        budget = max(budget, min_budget)
    return min(budget, remaining)


def compute_attempt_and_verify_deadlines(
    *,
    now: float,
    overall_deadline: float,
    reserve_fraction: float,
    reserve_cap_s: float,
    reserve_min_s: float,
) -> tuple[float, float]:
    """Return (attempt_deadline, overall_deadline).

    attempt_deadline is set earlier to keep a reserve for verification.
    Verification is allowed to use time until overall_deadline.
    """

    remaining = max(0.0, float(overall_deadline) - float(now))
    if remaining <= 0:
        return float(now), float(overall_deadline)

    frac = max(0.0, min(0.95, float(reserve_fraction)))
    cap = max(0.0, float(reserve_cap_s))
    min_s = max(0.0, float(reserve_min_s))

    reserve = remaining * frac
    if cap > 0:
        reserve = min(reserve, cap)
    if min_s > 0:
        reserve = max(reserve, min_s)

    # Never reserve more than what we have.
    reserve = min(reserve, remaining)
    attempt_deadline = float(overall_deadline) - reserve
    # Clamp to [now, overall_deadline]
    attempt_deadline = max(float(now), min(float(overall_deadline), attempt_deadline))
    return attempt_deadline, float(overall_deadline)
