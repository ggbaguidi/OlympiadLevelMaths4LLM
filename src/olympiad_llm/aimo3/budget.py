from __future__ import annotations

"""Time budget helpers for AIMO3.

The common failure mode in Kaggle is spending the entire per-problem budget on
exploration attempts, leaving too little time for second-stage verification.

We address this by reserving a small slice of time for verification.
"""


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
