from olympiad_llm.aimo3.attempts import AttemptResult, AttemptStats
from olympiad_llm.aimo3.solver import AIMO3Solver


def test_entropy_score_breaks_ties_when_other_signals_equal():
    # Both answers have:
    # - 1 vote
    # - 1 verified attempt
    # So entropy_score should act as a tie-breaker: lower entropy => higher score.
    results = [
        AttemptResult(attempt=1, answer=1, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0, mean_entropy=2.0)),
        AttemptResult(attempt=2, answer=2, stats=AttemptStats(token_count=50, python_calls=1, python_errors=0, mean_entropy=0.5)),
    ]

    ranked = AIMO3Solver._rank_answers(results)  # noqa: SLF001
    assert ranked
    assert ranked[0][0] == 2
