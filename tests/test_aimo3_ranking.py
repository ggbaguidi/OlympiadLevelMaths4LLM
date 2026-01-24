from olympiad_llm.aimo3.attempts import AttemptResult, AttemptStats
from olympiad_llm.aimo3.solver import AIMO3Solver


def test_rank_answers_counts_verified():
    # We can test the pure ranking function without constructing the solver.
    results = [
        AttemptResult(attempt=1, answer=10, stats=AttemptStats(token_count=100, python_calls=1, python_errors=0)),
        AttemptResult(attempt=2, answer=10, stats=AttemptStats(token_count=90, python_calls=0, python_errors=0)),
        AttemptResult(attempt=3, answer=11, stats=AttemptStats(token_count=80, python_calls=2, python_errors=1)),
    ]
    ranked = AIMO3Solver._rank_answers(results)  # noqa: SLF001
    assert ranked
    # Answer 10 has 2 votes and 1 verified attempt.
    assert ranked[0][0] == 10
    assert ranked[0][1]["votes"] == 2
    assert ranked[0][1]["verified"] == 1


def test_rank_prefers_verified_over_more_votes():
    # Verified-first: a candidate with a clean tool run should outrank
    # a higher-vote candidate with no clean tool runs.
    results = [
        AttemptResult(attempt=1, answer=1, stats=AttemptStats(token_count=50, python_calls=0, python_errors=0)),
        AttemptResult(attempt=2, answer=1, stats=AttemptStats(token_count=60, python_calls=0, python_errors=0)),
        AttemptResult(attempt=3, answer=1, stats=AttemptStats(token_count=70, python_calls=1, python_errors=1)),
        AttemptResult(attempt=4, answer=2, stats=AttemptStats(token_count=80, python_calls=1, python_errors=0)),
    ]

    ranked = AIMO3Solver._rank_answers(results)  # noqa: SLF001
    assert ranked
    assert ranked[0][0] == 2
    assert ranked[0][1]["verified"] == 1
