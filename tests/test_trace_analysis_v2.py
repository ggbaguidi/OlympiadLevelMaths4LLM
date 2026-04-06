import json
from pathlib import Path

from olympiad_llm.aimo3.v2.trace_analysis import analyze_trace, build_attempt_summaries


def test_detects_verified_filter_failure_signature(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    events = [
        {
            "event": "attempt_end",
            "problem_id": "p1",
            "answer": 160,
            "python_calls": 12,
            "python_errors": 1,
        },
        {
            "event": "attempt_end",
            "problem_id": "p1",
            "answer": 160,
            "python_calls": 8,
            "python_errors": 2,
        },
        {
            "event": "attempt_end",
            "problem_id": "p1",
            "answer": 266,
            "python_calls": 4,
            "python_errors": 0,
        },
        {"event": "solve_end", "problem_id": "p1", "answer": 266, "time_s": 1.2},
    ]
    trace_file.write_text(
        "\n".join(json.dumps(ev) for ev in events) + "\n",
        encoding="utf-8",
    )

    analyses = analyze_trace(trace_file)
    assert len(analyses) == 1
    a = analyses[0]
    assert a.problem_id == "p1"
    assert a.chosen == 266
    assert a.top_vote_answer == 160
    assert a.top_vote_count == 2
    assert a.chosen_vote_count == 1
    assert a.likely_verified_filter_failure is True


def test_parses_early_exit_reason_from_attempt_trace(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    events = [
        {
            "event": "attempt_end",
            "problem_id": "p2",
            "attempt": 1,
            "answer": 77,
            "python_calls": 1,
            "python_errors": 0,
            "early_exit_reason": "tool_final_marker",
            "extraction_rule": "tool_final_marker",
        },
        {"event": "solve_end", "problem_id": "p2", "answer": 77, "time_s": 0.5},
    ]
    trace_file.write_text(
        "\n".join(json.dumps(ev) for ev in events) + "\n",
        encoding="utf-8",
    )

    attempts_by_pid = build_attempt_summaries(trace_file)
    assert attempts_by_pid["p2"][0].early_exit_reason == "tool_final_marker"
