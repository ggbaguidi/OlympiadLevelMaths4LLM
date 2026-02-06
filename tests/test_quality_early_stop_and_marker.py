from olympiad_llm.aimo3.attempts import AttemptResult, AttemptStats
from olympiad_llm.aimo3.config import AIMO3Config
from olympiad_llm.aimo3.solver import AIMO3Solver


def test_has_verification_marker():
    assert (
        AIMO3Solver._has_verification_marker("abc VERIFIED_OK def", "VERIFIED_OK")
        is True
    )  # noqa: SLF001
    assert (
        AIMO3Solver._has_verification_marker("abc", "VERIFIED_OK") is False
    )  # noqa: SLF001
    assert (
        AIMO3Solver._has_verification_marker(None, "VERIFIED_OK") is False
    )  # noqa: SLF001


def test_quality_early_stop_requires_verified_by_default():
    s = AIMO3Solver.__new__(AIMO3Solver)
    s.cfg = AIMO3Config(early_stop=3, early_stop_min_verified=1)

    # Popular answer but no clean tool runs => should NOT stop early.
    detailed = [
        AttemptResult(
            attempt=1,
            answer=7,
            stats=AttemptStats(token_count=10, python_calls=0, python_errors=0),
        ),
        AttemptResult(
            attempt=2,
            answer=7,
            stats=AttemptStats(token_count=10, python_calls=0, python_errors=0),
        ),
        AttemptResult(
            attempt=3,
            answer=7,
            stats=AttemptStats(token_count=10, python_calls=0, python_errors=0),
        ),
    ]
    assert s._should_early_stop(detailed) is False  # noqa: SLF001

    # Add one clean tool run => now we can stop early.
    detailed.append(
        AttemptResult(
            attempt=4,
            answer=7,
            stats=AttemptStats(token_count=10, python_calls=1, python_errors=0),
        )
    )
    assert s._should_early_stop(detailed) is True  # noqa: SLF001


def test_quality_early_stop_can_be_vote_only():
    s = AIMO3Solver.__new__(AIMO3Solver)
    s.cfg = AIMO3Config(early_stop=3, early_stop_min_verified=0)

    detailed = [
        AttemptResult(attempt=1, answer=1, stats=AttemptStats()),
        AttemptResult(attempt=2, answer=1, stats=AttemptStats()),
        AttemptResult(attempt=3, answer=1, stats=AttemptStats()),
    ]
    assert s._should_early_stop(detailed) is True  # noqa: SLF001
