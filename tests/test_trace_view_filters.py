import json
from pathlib import Path

from olympiad_llm.aimo3.trace_view import iter_attempt_transcripts


def _write_jsonl(path: Path, events: list[dict]) -> None:
    lines = [json.dumps(ev, ensure_ascii=False) for ev in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_trace_view_filter_by_attempt_and_answer(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            {
                "event": "attempt_end",
                "problem_id": "p1",
                "attempt": 1,
                "answer": 42,
                "python_calls": 1,
                "python_errors": 0,
            },
            {
                "event": "attempt_end",
                "problem_id": "p1",
                "attempt": 2,
                "answer": "43",
                "python_calls": 2,
                "python_errors": 0,
                "lean_calls": "3",
            },
            {
                "event": "attempt_end",
                "problem_id": "p2",
                "attempt": 2,
                "answer": 43,
                "python_calls": 9,
                "python_errors": 1,
            },
            {"event": "other", "problem_id": "p1", "attempt": 99, "answer": 999},
        ],
    )

    # Filter by attempt only
    got = list(iter_attempt_transcripts(p, problem_id="p1", attempt=2))
    assert len(got) == 1
    assert got[0].problem_id == "p1"
    assert got[0].attempt == 2
    assert got[0].answer == 43
    assert got[0].lean_calls == 3

    # Filter by answer only
    got = list(iter_attempt_transcripts(p, problem_id="p1", answer=42))
    assert len(got) == 1
    assert got[0].attempt == 1

    # Filter by both answer and attempt
    got = list(iter_attempt_transcripts(p, problem_id="p1", attempt=2, answer=43))
    assert len(got) == 1
    assert got[0].attempt == 2

    # Mismatch should return nothing
    got = list(iter_attempt_transcripts(p, problem_id="p1", attempt=1, answer=43))
    assert got == []
