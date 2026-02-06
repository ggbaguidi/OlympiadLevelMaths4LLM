from olympiad_llm.aimo3.attempts import AttemptResult, AttemptStats
from olympiad_llm.aimo3.config import AIMO3Config


def test_truncate_behavior():
    from olympiad_llm.aimo3.solver import AIMO3Solver

    assert AIMO3Solver._truncate("hello", 10) == "hello"
    assert AIMO3Solver._truncate("hello", 0) == ""
    assert AIMO3Solver._truncate("abcdef", 3).endswith("def")


def test_attempt_row_formatting():
    from olympiad_llm.aimo3.solver import AIMO3Solver

    s = AIMO3Solver.__new__(AIMO3Solver)
    s.cfg = AIMO3Config(display_attempt_text_chars=5, capture_attempt_text_chars=10)
    r = AttemptResult(
        attempt=1,
        answer=7,
        stats=AttemptStats(token_count=12, python_calls=1, python_errors=0),
        output_text="0123456789",
    )
    row = s._attempt_to_row(r)
    assert row["Answer"] == 7
    assert row["ToolVerified"] is True
    assert "Snippet" in row
