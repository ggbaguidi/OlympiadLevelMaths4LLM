from __future__ import annotations

"""General recovery heuristics.

These utilities are intentionally problem-agnostic: they only look at runtime
signals (tool errors, repeated failures) and decide whether to abort an attempt
or recycle a sandbox.
"""

from dataclasses import dataclass

from .attempts import AttemptResult


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


def should_schedule_recovery_attempt(
    *,
    result: AttemptResult,
    remaining_s: float,
    recovery_trigger_python_errors: int,
    recovery_min_remaining_s: float,
) -> bool:
    """Return True if we should schedule a follow-up recovery attempt.

    This is intentionally problem-agnostic.
    """

    if float(remaining_s) < float(recovery_min_remaining_s):
        return False

    # If we already got a valid answer, don't schedule a recovery.
    if isinstance(result.answer, int):
        return False

    tag = str(getattr(result, "tag", "") or "")
    if "tool_abort" in tag:
        return True

    pe = int(result.stats.python_errors)
    if int(recovery_trigger_python_errors) > 0 and pe >= int(recovery_trigger_python_errors):
        return True

    return False


def tool_call_cap_for_attempt(*, attempt_tag: str | None, recovery_micro_cap: int) -> int | None:
    """Return a max python tool-call cap for this attempt.

    - None: no cap enforcement
    - 0: tool disabled

    This is primarily used to enforce recovery variants, without relying on
    tool-config wiring.
    """

    tag = str(attempt_tag or "")
    if "variant=no_tool" in tag:
        return 0
    if "variant=micro_tool" in tag:
        return max(0, int(recovery_micro_cap))
    return None
