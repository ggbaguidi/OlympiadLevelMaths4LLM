from __future__ import annotations

"""Utilities for reading per-attempt transcripts from AIMO-3 trace JSONL.

This is intentionally stdlib-only so it works in Kaggle/offline environments.

We record attempt transcripts as events with event=="attempt_end".
These transcripts intentionally exclude hidden analysis/CoT.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class AttemptTranscript:
    problem_id: str
    attempt: int
    tag: str | None
    answer: int | None
    token_count: int | None
    python_calls: int | None
    python_errors: int | None
    assistant_final: str | None
    assistant_commentary: str | None
    python_calls_text: list[str] | None
    python_outputs_text: list[str] | None


def _iter_events(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def iter_attempt_transcripts(path: str | Path, problem_id: str | None = None) -> Iterable[AttemptTranscript]:
    for ev in _iter_events(path):
        ev_name = ev.get("event") or ev.get("event_type")
        if ev_name != "attempt_end":
            continue
        pid = ev.get("problem_id")
        if not isinstance(pid, str) or not pid:
            continue
        if problem_id is not None and pid != problem_id:
            continue
        yield AttemptTranscript(
            problem_id=pid,
            attempt=int(ev.get("attempt") or 0),
            tag=ev.get("tag"),
            answer=ev.get("answer"),
            token_count=ev.get("token_count"),
            python_calls=ev.get("python_calls"),
            python_errors=ev.get("python_errors"),
            assistant_final=ev.get("assistant_final"),
            assistant_commentary=ev.get("assistant_commentary"),
            python_calls_text=ev.get("python_calls_text"),
            python_outputs_text=ev.get("python_outputs_text"),
        )


def _print_attempt(a: AttemptTranscript) -> None:
    header = f"problem_id={a.problem_id} attempt={a.attempt} answer={a.answer} calls={a.python_calls} errors={a.python_errors} tag={a.tag}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    if a.assistant_commentary:
        print("\n--- assistant_commentary ---\n")
        print(a.assistant_commentary)

    if a.assistant_final:
        print("\n--- assistant_final ---\n")
        print(a.assistant_final)

    if a.python_calls_text:
        print("\n--- python_calls ---\n")
        for i, s in enumerate(a.python_calls_text, start=1):
            print(f"[call {i}]\n{s}\n")

    if a.python_outputs_text:
        print("\n--- python_outputs ---\n")
        for i, s in enumerate(a.python_outputs_text, start=1):
            print(f"[output {i}]\n{s}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="View AIMO-3 per-attempt transcripts from a trace JSONL.")
    ap.add_argument("--path", default="aimo3_trace.jsonl", help="Path to the trace JSONL")
    ap.add_argument("--problem-id", default=None, help="Filter to a specific problem_id")
    ap.add_argument("--max", type=int, default=50, help="Max attempts to print")
    args = ap.parse_args(argv)

    n = 0
    for a in iter_attempt_transcripts(args.path, problem_id=args.problem_id):
        _print_attempt(a)
        n += 1
        if n >= int(args.max):
            break

    if n == 0:
        print("No attempt_end events found (did you enable AIMO3_TRACE and AIMO3_TRACE_ATTEMPTS?).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
