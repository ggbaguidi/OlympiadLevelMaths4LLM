from __future__ import annotations

"""Time budget helpers for AIMO3.

The common failure mode in Kaggle is spending the entire per-problem budget on
exploration attempts, leaving too little time for second-stage verification.

We address this by reserving a small slice of time for verification.
"""


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
