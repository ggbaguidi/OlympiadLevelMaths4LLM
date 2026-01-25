import json


def test_trace_recorder_writes_jsonl(tmp_path):
    from olympiad_llm.aimo3.trace import TraceRecorder

    p = tmp_path / "trace.jsonl"
    tr = TraceRecorder(enabled=True, path=str(p), include_problem_text=False)
    tr.record({"event": "x", "problem": None, "value": 1})

    lines = p.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["event"] == "x"
    assert obj["event_type"] == "x"
    assert obj["value"] == 1
    assert "ts" in obj


def test_trace_recorder_disabled_no_file(tmp_path):
    from olympiad_llm.aimo3.trace import TraceRecorder

    p = tmp_path / "trace.jsonl"
    tr = TraceRecorder(enabled=False, path=str(p), include_problem_text=False)
    tr.record({"event": "x"})
    assert not p.exists()
