from olympiad_llm.aimo3.recovery import ToolRecoveryPolicy, should_abort_attempt, should_recycle_sandbox


def test_should_abort_attempt_total_errors():
    policy = ToolRecoveryPolicy(abort_after_python_errors=2, abort_after_consecutive_python_errors=0)
    assert not should_abort_attempt(python_errors=1, consecutive_python_errors=1, policy=policy)
    assert should_abort_attempt(python_errors=2, consecutive_python_errors=0, policy=policy)


def test_should_abort_attempt_consecutive_errors():
    policy = ToolRecoveryPolicy(abort_after_python_errors=0, abort_after_consecutive_python_errors=3)
    assert not should_abort_attempt(python_errors=10, consecutive_python_errors=2, policy=policy)
    assert should_abort_attempt(python_errors=2, consecutive_python_errors=3, policy=policy)


def test_should_recycle_sandbox_on_exception():
    policy = ToolRecoveryPolicy(recycle_sandbox_after_python_errors=100)
    assert should_recycle_sandbox(python_errors=0, had_exception=True, policy=policy)


def test_should_recycle_sandbox_on_many_errors():
    policy = ToolRecoveryPolicy(recycle_sandbox_after_python_errors=4)
    assert not should_recycle_sandbox(python_errors=3, had_exception=False, policy=policy)
    assert should_recycle_sandbox(python_errors=4, had_exception=False, policy=policy)
