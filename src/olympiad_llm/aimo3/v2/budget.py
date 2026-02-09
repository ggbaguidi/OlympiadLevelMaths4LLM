# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""Time budget helpers for AIMO3.

The common failure mode in Kaggle is spending the entire per-problem budget on
exploration attempts, leaving too little time for second-stage verification.

We address this by reserving a small slice of time for verification.

Additionally, we support adaptive extension: if a problem shows "hardness signals"
(no consensus, high variance), we can draw extra time from a flex pool banked
from easy problems that solved quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TimeBudgetTracker:
    """Track actual time usage and dynamically adjust per-problem budgets.

    Key insight: if early problems solve quickly, we bank that time for harder problems.

    New in adaptive mode:
    - Reserve a "flex pool" (15% of total time) for hard problems
    - Extend budget mid-solve when hardness signals are detected
    - Cap extension per problem (max 2x base budget)
    """

    total_budget_s: float
    total_problems: int
    base_timeout_s: float = 300.0
    high_timeout_s: float = 900.0
    # Minimum budget per problem (never go below this)
    min_budget_s: float = 120.0

    # Adaptive extension settings
    # Fraction of total time reserved as flex pool for hard problems
    flex_pool_fraction: float = 0.15
    # Max extension multiplier (e.g., 2.0 = can double the base budget)
    max_extension_multiplier: float = 2.0
    # Trigger extension if no consensus after this fraction of base budget spent
    hardness_trigger_fraction: float = 0.5
    # Min distinct answers to consider "no consensus"
    hardness_min_distinct_answers: int = 3
    # Aggressive early-stop threshold: if we get consensus in < this fraction of budget, bank savings
    easy_problem_threshold_fraction: float = 0.3

    # State tracking
    problems_solved: int = field(default=0, init=False)
    total_time_used_s: float = field(default=0.0, init=False)
    solve_times: List[float] = field(default_factory=list, init=False)
    flex_pool_used_s: float = field(default=0.0, init=False)
    extensions_granted: int = field(default=0, init=False)

    @property
    def problems_remaining(self) -> int:
        return max(0, self.total_problems - self.problems_solved)

    @property
    def time_remaining_s(self) -> float:
        return max(0.0, self.total_budget_s - self.total_time_used_s)

    @property
    def avg_solve_time_s(self) -> float:
        """Rolling average of actual solve times."""
        if not self.solve_times:
            return self.base_timeout_s
        return sum(self.solve_times) / len(self.solve_times)

    @property
    def flex_pool_total_s(self) -> float:
        """Total flex pool size based on config fraction."""
        return self.total_budget_s * self.flex_pool_fraction

    @property
    def flex_pool_remaining_s(self) -> float:
        """Remaining flex pool time."""
        return max(0.0, self.flex_pool_total_s - self.flex_pool_used_s)

    def compute_budget(self) -> float:
        """Compute the budget for the next problem.

        Core logic: subtract *all* time already used from the total, set aside
        the flex-pool reserve, then divide the rest equally among the remaining
        problems.  This guarantees that every second spent is accounted for
        before a new budget is assigned.
        """
        if self.problems_remaining <= 0:
            return 0.0

        # 1. True remaining time (total - everything spent so far).
        remaining = self.time_remaining_s

        # 2. Set aside flex reserve for hard-problem extensions.
        flex_reserve = min(self.flex_pool_remaining_s, remaining)
        available = max(0.0, remaining - flex_reserve)

        # 3. Divide evenly among remaining problems.
        budget = available / self.problems_remaining

        # 4. Apply bounds.
        budget = max(self.min_budget_s, budget)
        budget = min(self.high_timeout_s, budget)

        # 5. Never exceed total remaining time (flex included as absolute ceiling).
        budget = min(budget, remaining)

        return budget

    def request_extension(
        self,
        *,
        time_spent_s: float,
        current_budget_s: float,
        n_distinct_answers: int,
        has_consensus: bool,
    ) -> float:
        """Request a budget extension for a hard problem.

        Returns the additional time granted (0 if no extension warranted).

        Hardness signals:
        - Spent significant time (> hardness_trigger_fraction of budget)
        - No consensus (many distinct answers or no consensus flag)
        - Flex pool has remaining capacity
        """
        # Check if we've spent enough time to judge hardness
        trigger_threshold = current_budget_s * self.hardness_trigger_fraction
        if time_spent_s < trigger_threshold:
            return 0.0

        # Check for hardness signals
        is_hard = (
            not has_consensus
            or n_distinct_answers >= self.hardness_min_distinct_answers
        )

        if not is_hard:
            return 0.0

        # Calculate max allowed extension
        max_extension = current_budget_s * (self.max_extension_multiplier - 1.0)

        # True remaining time accounting for in-flight spend on THIS problem.
        true_remaining = max(0.0, self.time_remaining_s - time_spent_s)

        available_extension = min(
            max_extension,
            self.flex_pool_remaining_s,
            true_remaining,
        )

        if available_extension <= 10.0:  # Not worth extending for < 10s
            return 0.0

        # Grant extension (draw from flex pool)
        extension = min(available_extension, max_extension)
        self.flex_pool_used_s += extension
        self.extensions_granted += 1

        return extension

    def request_no_answer_extension(
        self,
        *,
        time_spent_s: float,
        current_budget_s: float,
    ) -> float:
        """Request a budget extension when ALL attempts returned None.

        This is the strongest hardness signal: the model couldn't produce any
        valid answer at all.  We draw aggressively from the flex pool (up to
        the full max_extension_multiplier) to give the solver a second chance.

        Returns the additional seconds granted (0 if nothing available).
        """
        max_extension = current_budget_s * (self.max_extension_multiplier - 1.0)

        # True remaining time accounting for in-flight spend on THIS problem.
        true_remaining = max(0.0, self.time_remaining_s - time_spent_s)

        available_extension = min(
            max_extension,
            self.flex_pool_remaining_s,
            true_remaining,
        )

        if available_extension <= 10.0:
            return 0.0

        extension = min(available_extension, max_extension)
        self.flex_pool_used_s += extension
        self.extensions_granted += 1
        return extension

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
            f"Remaining: {self.time_remaining_s:.0f}s | "
            f"Flex: {self.flex_pool_remaining_s:.0f}s/{self.flex_pool_total_s:.0f}s | "
            f"Avg: {self.avg_solve_time_s:.0f}s | "
            f"Next: {self.compute_budget():.0f}s | "
            f"Extensions: {self.extensions_granted}"
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
