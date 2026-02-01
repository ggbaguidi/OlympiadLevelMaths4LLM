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


def test_rank_does_not_let_tag_diversity_overpower_votes_when_verified_ties():
    # Both answers have verified support, but answer=1 has stronger vote support.
    # Ensure tag diversity acts as a tie-breaker, not the primary driver.
    results = [
        AttemptResult(attempt=1, answer=1, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0), tag="a"),
        AttemptResult(attempt=2, answer=1, stats=AttemptStats(token_count=50, python_calls=0, python_errors=0), tag="a"),
        AttemptResult(attempt=3, answer=1, stats=AttemptStats(token_count=50, python_calls=0, python_errors=0), tag="a"),
        # answer=2 appears under two tags, but only has 2 votes total.
        AttemptResult(attempt=4, answer=2, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0), tag="x"),
        AttemptResult(attempt=5, answer=2, stats=AttemptStats(token_count=50, python_calls=0, python_errors=0), tag="y"),
    ]
    ranked = AIMO3Solver._rank_answers(results)  # noqa: SLF001
    assert ranked
    assert ranked[0][0] == 1


def test_rank_penalizes_timeout_attempts():
    """Answers from timed-out attempts should be ranked lower than equal-vote alternatives."""
    from olympiad_llm.aimo3.ranking import rank_candidates

    # Answer 1: 2 votes, 1 verified, 1 timeout
    # Answer 2: 2 votes, 1 verified, 0 timeouts
    # Both have same votes and verified count, but answer 2 has no timeouts.
    results = [
        AttemptResult(attempt=1, answer=1, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0, timeout_count=0)),
        AttemptResult(attempt=2, answer=1, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0, timeout_count=1)),
        AttemptResult(attempt=3, answer=2, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0, timeout_count=0)),
        AttemptResult(attempt=4, answer=2, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0, timeout_count=0)),
    ]
    ranked = rank_candidates(results)
    assert ranked
    # Answer 2 should rank higher because it has no timeouts.
    assert ranked[0][0] == 2
    assert ranked[0][1]["timeout_attempts"] == 0
    assert ranked[1][0] == 1
    assert ranked[1][1]["timeout_attempts"] == 1


def test_rank_magnitude_aware_boosts_outliers():
    """When answers span wildly different magnitudes, boost large outliers.

    This prevents picking small "easy wrong" answers (e.g., 15) when the true
    answer is much larger (e.g., 8687). The outlier (e.g., 11549) should rank higher.
    """
    from olympiad_llm.aimo3.ranking import rank_candidates

    # Simulate: many small answers (1-20), one large outlier (11549)
    # Small answers 15, 16 are "verified", outlier 11549 is not
    results = [
        AttemptResult(attempt=0, answer=14, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=1, answer=12, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=2, answer=11549, stats=AttemptStats(python_calls=5, python_errors=1)),  # Outlier!
        AttemptResult(attempt=3, answer=11, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=4, answer=15, stats=AttemptStats(python_calls=5, python_errors=0)),  # Verified
        AttemptResult(attempt=5, answer=16, stats=AttemptStats(python_calls=5, python_errors=0)),  # Verified
        AttemptResult(attempt=6, answer=13, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=7, answer=5, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=8, answer=6, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=9, answer=3, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=10, answer=1, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=11, answer=4, stats=AttemptStats(python_calls=5, python_errors=1)),
    ]

    # With magnitude awareness, outlier should win
    ranked_mag = rank_candidates(results, magnitude_aware=True)
    assert ranked_mag[0][0] == 11549, f"Expected 11549 to be top, got {ranked_mag[0][0]}"

    # Without magnitude awareness, verified small answers would win
    ranked_no_mag = rank_candidates(results, magnitude_aware=False)
    assert ranked_no_mag[0][0] in (15, 16), f"Expected 15 or 16, got {ranked_no_mag[0][0]}"


def test_rank_magnitude_aware_no_effect_when_similar_magnitudes():
    """Magnitude awareness should not affect ranking when all answers are similar."""
    from olympiad_llm.aimo3.ranking import rank_candidates

    # All answers in same magnitude range (10-30)
    results = [
        AttemptResult(attempt=0, answer=10, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=1, answer=20, stats=AttemptStats(python_calls=5, python_errors=0)),  # Verified
        AttemptResult(attempt=2, answer=15, stats=AttemptStats(python_calls=5, python_errors=1)),
        AttemptResult(attempt=3, answer=25, stats=AttemptStats(python_calls=5, python_errors=1)),
    ]

    ranked = rank_candidates(results, magnitude_aware=True)
    # Verified answer (20) should still win - no magnitude outliers
    assert ranked[0][0] == 20

