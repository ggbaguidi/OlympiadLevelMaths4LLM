from olympiad_llm.aimo3.trace_analysis import summarize_trace


def test_summarize_trace_basic_parsing_and_risk():
    events = [
        {
            "event": "solve_start",
            "problem_id": "abc",
            "budget_s": 100.0,
            "attempt_deadline_in_s": 80.0,
            "overall_deadline_in_s": 100.0,
        },
        {
            "event": "solve_end",
            "problem_id": "abc",
            "status": "ok",
            "chosen": 7,
            "elapsed_s": 12.0,
            "attempts": [
                {
                    "attempt": 1,
                    "answer": 7,
                    "python_calls": 2,
                    "python_errors": 0,
                    "tag": "standard",
                },
                {
                    "attempt": 2,
                    "answer": 8,
                    "python_calls": 0,
                    "python_errors": 0,
                    "tag": "analytic",
                },
            ],
            "decision": {
                "ranked": [
                    {"answer": 7, "votes": 1, "verified": 1, "tag_diversity": 1},
                    {"answer": 8, "votes": 1, "verified": 0, "tag_diversity": 1},
                ],
                "second_stage": None,
            },
        },
    ]

    s = summarize_trace(events)
    assert len(s) == 1
    ps = s[0]
    assert ps.problem_id == "abc"
    assert ps.status == "ok"
    assert ps.chosen == 7
    assert ps.n_attempts == 2
    assert ps.n_valid_attempts == 2
    assert ps.n_verified_attempts == 1
    assert ps.top_verified == 1
    assert ps.risk_score < 3.0


def test_summarize_trace_marks_unverified_as_riskier():
    events = [
        {
            "event": "solve_end",
            "problem_id": "p1",
            "status": "ok",
            "chosen": 1,
            "elapsed_s": 1.0,
            "attempts": [],
            "decision": {
                "ranked": [{"answer": 1, "votes": 1, "verified": 0, "tag_diversity": 1}]
            },
        },
        {
            "event": "solve_end",
            "problem_id": "p2",
            "status": "ok",
            "chosen": 2,
            "elapsed_s": 1.0,
            "attempts": [],
            "decision": {
                "ranked": [{"answer": 2, "votes": 1, "verified": 1, "tag_diversity": 1}]
            },
        },
    ]

    s = summarize_trace(events)
    # Highest risk first
    assert s[0].problem_id == "p1"
    assert s[0].risk_score > s[1].risk_score


def test_summarize_trace_accepts_event_type_key():
    events = [
        {"event_type": "solve_start", "problem_id": "abc", "budget_s": 10.0},
        {
            "event_type": "solve_end",
            "problem_id": "abc",
            "status": "ok",
            "chosen": 1,
            "elapsed_s": 1.0,
            "attempts": [],
            "decision": {
                "ranked": [{"answer": 1, "votes": 1, "verified": 1, "tag_diversity": 1}]
            },
        },
    ]
    s = summarize_trace(events)
    assert len(s) == 1
    assert s[0].problem_id == "abc"
