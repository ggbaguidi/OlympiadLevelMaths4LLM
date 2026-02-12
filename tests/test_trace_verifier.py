import json
from pathlib import Path


def analyze_trace(
    trace_path: Path, ground_truth: dict[str, int]
) -> list[tuple[str, int, int]]:
    """Return a list of (problem_id, answer, gt) for solve_end events that have a ground truth."""
    mismatches: list[tuple[str, int, int]] = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ev = obj.get("event") or obj.get("event_type")
            if ev != "solve_end":
                continue
            pid = obj.get("problem_id")
            ans = obj.get("answer")
            if pid is None or ans is None:
                continue
            if pid in ground_truth:
                gt = ground_truth[pid]
                mismatches.append((pid, int(ans), int(gt)))
    return mismatches


def test_detects_mismatch(tmp_path: Path) -> None:
    # Create a tiny synthetic trace with a solve_end entry that disagrees with ground truth.
    trace_file = tmp_path / "small_trace.jsonl"
    entry = {
        "event": "solve_end",
        "problem_id": "test_0",
        "answer": 1,
        "time_s": 1.23,
        "attempts_total": 3,
    }
    trace_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    gt = {"test_0": 57447}

    mismatches = analyze_trace(trace_file, gt)

    # We expect one mismatch and the tuple to contain the recorded answer and the ground truth.
    assert len(mismatches) == 1
    pid, ans, gval = mismatches[0]
    assert pid == "test_0"
    assert ans == 1
    assert gval == 57447
