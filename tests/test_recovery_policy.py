from olympiad_llm.aimo3.recovery import (
    ToolRecoveryPolicy,
    should_abort_attempt,
    should_recycle_sandbox,
)
from olympiad_llm.aimo3.recovery import should_schedule_recovery_attempt
from olympiad_llm.aimo3.recovery import tool_call_cap_for_attempt
from olympiad_llm.aimo3.recovery import should_schedule_format_recovery_attempt
from olympiad_llm.aimo3.attempts import AttemptResult, AttemptStats


def test_should_abort_attempt_total_errors():
    policy = ToolRecoveryPolicy(
        abort_after_python_errors=2, abort_after_consecutive_python_errors=0
    )
    assert not should_abort_attempt(
        python_errors=1, consecutive_python_errors=1, policy=policy
    )
    assert should_abort_attempt(
        python_errors=2, consecutive_python_errors=0, policy=policy
    )


def test_should_abort_attempt_consecutive_errors():
    policy = ToolRecoveryPolicy(
        abort_after_python_errors=0, abort_after_consecutive_python_errors=3
    )
    assert not should_abort_attempt(
        python_errors=10, consecutive_python_errors=2, policy=policy
    )
    assert should_abort_attempt(
        python_errors=2, consecutive_python_errors=3, policy=policy
    )


def test_should_abort_attempt_on_timeouts():
    """Abort attempt when too many tool calls time out."""
    policy = ToolRecoveryPolicy(
        abort_after_python_errors=0,
        abort_after_consecutive_python_errors=0,
        abort_after_timeouts=2,
    )
    assert not should_abort_attempt(
        python_errors=0, consecutive_python_errors=0, timeout_count=1, policy=policy
    )
    assert should_abort_attempt(
        python_errors=0, consecutive_python_errors=0, timeout_count=2, policy=policy
    )


def test_should_recycle_sandbox_on_exception():
    policy = ToolRecoveryPolicy(recycle_sandbox_after_python_errors=100)
    assert should_recycle_sandbox(python_errors=0, had_exception=True, policy=policy)


def test_should_recycle_sandbox_on_many_errors():
    policy = ToolRecoveryPolicy(recycle_sandbox_after_python_errors=4)
    assert not should_recycle_sandbox(
        python_errors=3, had_exception=False, policy=policy
    )
    assert should_recycle_sandbox(python_errors=4, had_exception=False, policy=policy)


def test_should_schedule_recovery_attempt_requires_time_and_no_answer():
    r = AttemptResult(
        attempt=1,
        answer=None,
        stats=AttemptStats(token_count=0, python_calls=1, python_errors=3),
        tag="x",
    )
    assert not should_schedule_recovery_attempt(
        result=r,
        remaining_s=5.0,
        recovery_trigger_python_errors=3,
        recovery_min_remaining_s=10.0,
    )


def test_should_schedule_recovery_attempt_triggers_on_tool_abort_tag():
    r = AttemptResult(
        attempt=1,
        answer=None,
        stats=AttemptStats(token_count=0, python_calls=1, python_errors=1),
        tag="x|tool_abort",
    )
    assert should_schedule_recovery_attempt(
        result=r,
        remaining_s=30.0,
        recovery_trigger_python_errors=10,
        recovery_min_remaining_s=10.0,
    )


def test_should_schedule_recovery_attempt_triggers_on_error_count():
    r = AttemptResult(
        attempt=1,
        answer=None,
        stats=AttemptStats(token_count=0, python_calls=3, python_errors=4),
        tag="x",
    )
    assert should_schedule_recovery_attempt(
        result=r,
        remaining_s=30.0,
        recovery_trigger_python_errors=3,
        recovery_min_remaining_s=10.0,
    )


def test_should_schedule_recovery_attempt_not_when_answer_present():
    r = AttemptResult(
        attempt=1,
        answer=123,
        stats=AttemptStats(token_count=0, python_calls=0, python_errors=0),
        tag="x|tool_abort",
    )
    assert not should_schedule_recovery_attempt(
        result=r,
        remaining_s=30.0,
        recovery_trigger_python_errors=1,
        recovery_min_remaining_s=10.0,
    )


def test_tool_call_cap_for_attempt_variants():
    assert tool_call_cap_for_attempt(attempt_tag=None, recovery_micro_cap=2) is None
    assert (
        tool_call_cap_for_attempt(
            attempt_tag="recovery|variant=no_tool", recovery_micro_cap=2
        )
        == 0
    )
    assert (
        tool_call_cap_for_attempt(
            attempt_tag="recovery|variant=micro_tool", recovery_micro_cap=2
        )
        == 2
    )


def test_should_schedule_format_recovery_attempt():
    r = AttemptResult(
        attempt=1,
        answer=None,
        stats=AttemptStats(token_count=2500, python_calls=0, python_errors=0),
        tag="x",
    )
    assert should_schedule_format_recovery_attempt(
        result=r, remaining_s=30.0, trigger_tokens=2000, min_remaining_s=10.0
    )

    # Too few tokens => don't schedule
    r2 = AttemptResult(
        attempt=1,
        answer=None,
        stats=AttemptStats(token_count=500, python_calls=0, python_errors=0),
        tag="x",
    )
    assert not should_schedule_format_recovery_attempt(
        result=r2, remaining_s=30.0, trigger_tokens=2000, min_remaining_s=10.0
    )

    # If there were tool errors, don't schedule format recovery (it's probably a tool instability case)
    r3 = AttemptResult(
        attempt=1,
        answer=None,
        stats=AttemptStats(token_count=2500, python_calls=2, python_errors=1),
        tag="x",
    )
    assert not should_schedule_format_recovery_attempt(
        result=r3, remaining_s=30.0, trigger_tokens=2000, min_remaining_s=10.0
    )
