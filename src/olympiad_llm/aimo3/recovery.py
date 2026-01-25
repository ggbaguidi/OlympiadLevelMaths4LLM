from __future__ import annotations

"""General recovery heuristics.

These utilities are intentionally problem-agnostic: they only look at runtime
signals (tool errors, repeated failures) and decide whether to abort an attempt
or recycle a sandbox.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolRecoveryPolicy:
    abort_after_python_errors: int = 4
    abort_after_consecutive_python_errors: int = 3
    recycle_sandbox_after_python_errors: int = 4


def should_abort_attempt(*, python_errors: int, consecutive_python_errors: int, policy: ToolRecoveryPolicy) -> bool:
    """Return True if an attempt should stop early due to tool instability."""

    pe = int(python_errors)
    ce = int(consecutive_python_errors)
    if int(policy.abort_after_python_errors) > 0 and pe >= int(policy.abort_after_python_errors):
        return True
    if int(policy.abort_after_consecutive_python_errors) > 0 and ce >= int(policy.abort_after_consecutive_python_errors):
        return True
    return False


def should_recycle_sandbox(*, python_errors: int, had_exception: bool, policy: ToolRecoveryPolicy) -> bool:
    """Return True if a sandbox should be closed and replaced.

    We consider a sandbox 'poisoned' if it yields many tool errors (kernel issues,
    broken state) or if the attempt handler itself raised.
    """

    if bool(had_exception):
        return True
    pe = int(python_errors)
    if int(policy.recycle_sandbox_after_python_errors) > 0 and pe >= int(policy.recycle_sandbox_after_python_errors):
        return True
    return False
