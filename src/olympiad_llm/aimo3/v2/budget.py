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
from typing import List, Optional
import os


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
    # State tracking
    problems_solved: int = field(default=0, init=False)
    total_time_used_s: float = field(default=0.0, init=False)
    solve_times: List[float] = field(default_factory=list, init=False)
    flex_pool_used_s: float = field(default=0.0, init=False)
    extensions_granted: int = field(default=0, init=False)
    # Strategy controls: can be set via env vars. Options:
    # - 'equal': use equal share of remaining time
    # - 'base': use configured base_timeout_s for every problem
    # - 'avg': use rolling average
    # - 'cumulative': add carryover from previous problems
    # - 'hybrid': take max(equal, avg, base) (default behavior)
    budget_strategy: str = field(default="hybrid")

    # Carryover pool: accumulates leftover time from problems solved under budget
    carryover_pool_s: float = field(default=0.0, init=False)
    # Whether to distribute carryover across remaining problems (True) or
    # apply to the next single problem only (False)
    cumulative_distribute: bool = field(default=True)

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

    def __post_init__(self) -> None:
        # Read optional environment overrides
        strat = os.getenv("AIMO3_BUDGET_STRATEGY")
        if strat:
            self.budget_strategy = strat.lower()

        base_env = os.getenv("AIMO3_BASE_TIMEOUT_S")
        if base_env:
            try:
                self.base_timeout_s = float(base_env)
            except Exception:
                pass

        carry_env = os.getenv("AIMO3_CARRYOVER_ENABLED")
        if carry_env is not None:
            # treat any non-empty value other than '0'/'false' as enabled
            self.carryover_pool_s = 0.0
            if carry_env.lower() in ("0", "false", "no"):
                self.cumulative_distribute = False

        dist_env = os.getenv("AIMO3_CUMULATIVE_DISTRIBUTE")
        if dist_env is not None:
            if dist_env.lower() in ("0", "false", "no"):
                self.cumulative_distribute = False
            else:
                self.cumulative_distribute = True

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

        # 3. Divide evenly among remaining problems (excluding any carryover
        # amount which we'll treat separately when using cumulative mode).
        # Subtract current carryover from available so it won't be double-counted.
        carry = max(0.0, self.carryover_pool_s)
        available_excl_carry = max(0.0, available - carry)
        equal_share = available_excl_carry / self.problems_remaining

        # 4. Strategy selection
        strat = (self.budget_strategy or "hybrid").lower()
        budget = equal_share

        if strat == "equal":
            budget = equal_share
        elif strat == "base":
            budget = self.base_timeout_s
        elif strat == "avg":
            budget = self.avg_solve_time_s
        elif strat == "cumulative":
            # Distribute carryover across remaining problems (or apply all to
            # next problem if distribution disabled).
            if carry <= 0.0:
                additional = 0.0
            elif self.cumulative_distribute:
                additional = carry / self.problems_remaining
            else:
                additional = carry

            # Use at least the equal share or base, then add the carryover
            budget = max(equal_share, self.base_timeout_s) + additional
            # Reserve the portion of carry that we're allocating now so it's
            # not repeatedly applied. We subtract the portion we intend to
            # allocate immediately.
            reserved = min(additional, self.carryover_pool_s)
            self.carryover_pool_s = max(0.0, self.carryover_pool_s - reserved)
        else:
            # hybrid: be conservative and use max of equal, avg and base
            budget = max(equal_share, self.avg_solve_time_s, self.base_timeout_s)

        # 5. Apply bounds.
        budget = max(self.min_budget_s, budget)
        budget = min(self.high_timeout_s, budget)

        # 6. Never exceed total remaining time (flex included as absolute ceiling).
        budget = min(budget, remaining)

        return budget

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

    def record_solve(self, time_used_s: float, allocated_budget_s: Optional[float] = None) -> None:  # type: ignore[override]
        """Record completion of a problem.

        If `allocated_budget_s` is provided and the problem used less than the
        allocated amount, the leftover is added to the carryover pool so it
        can be used (or distributed) to future problems when using the
        cumulative strategy.
        """
        # base bookkeeping
        self.problems_solved += 1
        self.total_time_used_s += time_used_s
        self.solve_times.append(time_used_s)
        # Keep rolling window of last 10 for average
        if len(self.solve_times) > 10:
            self.solve_times.pop(0)

        # accumulate leftover into carryover pool
        if allocated_budget_s is not None:
            leftover = max(0.0, allocated_budget_s - time_used_s)
            if leftover > 0.0:
                self.carryover_pool_s += leftover

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
