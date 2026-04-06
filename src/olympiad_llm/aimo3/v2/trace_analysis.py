# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""Analyze AIMO-3 v2 JSONL traces.

Usage:
    python -m olympiad_llm.aimo3.v2.trace_analysis tmp/aimo3_trace.jsonl
    python -m olympiad_llm.aimo3.v2.trace_analysis tmp/aimo3_trace.jsonl --attempts-summary
    python -m olympiad_llm.aimo3.v2.trace_analysis tmp/aimo3_trace.jsonl --problem-id <PID> --show-attempts

Optional ground-truth mapping:
    python -m olympiad_llm.aimo3.v2.trace_analysis tmp/aimo3_trace.jsonl --answers answers.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ProblemAnalysis:
    """Structured analysis of a single problem-solving run."""

    problem_id: str
    chosen: int | None
    time_s: float
    attempts_total: int
    answered_attempts: int
    distinct_answers: int
    top_vote_answer: int | None
    top_vote_count: int
    chosen_vote_count: int
    chosen_verified_count: int
    top_vote_verified_count: int
    ranking_top_answer: int | None
    ranking_top_votes: int
    ranking_top_verified: int
    likely_verified_filter_failure: bool


@dataclass
class AttemptSummary:
    problem_id: str
    attempt: int
    tag: str
    answer: int | None
    token_count: int
    python_calls: int
    python_errors: int
    timeout_count: int
    deadline_exceeded: bool
    is_verified: bool
    last_error: str | None
    python_calls_text: list[str]
    python_outputs_text: list[str]
    full_reasoning_text: str | None


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, dict):
                yield obj


def _as_int(x: Any) -> int | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x.is_integer():
            return int(x)
        return None
    try:
        s = str(x).strip()
        if not s:
            return None
        if "." in s:
            f = float(s)
            if f.is_integer():
                return int(f)
            return None
        return int(s)
    except Exception:  # noqa: BLE001
        return None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float(default)


def load_answers(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("answers must be a JSON object: {problem_id: answer}")
    out: dict[str, int] = {}
    for k, v in data.items():
        iv = _as_int(v)
        if iv is None:
            continue
        out[str(k)] = iv
    return out


def load_event_groups(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Load solve_end and attempt_end events grouped by problem_id."""
    solve_end_by_pid: dict[str, dict[str, Any]] = {}
    attempt_end_by_pid: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for ev in _iter_jsonl(path):
        event = ev.get("event") or ev.get("event_type") or ev.get("type")
        pid = str(ev.get("problem_id") or "")
        if not pid:
            continue
        if event == "solve_end":
            solve_end_by_pid[pid] = ev
        elif event == "attempt_end":
            attempt_end_by_pid[pid].append(ev)

    return solve_end_by_pid, attempt_end_by_pid


def analyze_trace(path: Path) -> list[ProblemAnalysis]:
    solve_end_by_pid, attempt_end_by_pid = load_event_groups(path)

    analyses: list[ProblemAnalysis] = []
    for pid, end in solve_end_by_pid.items():
        attempts = attempt_end_by_pid.get(pid, [])

        votes: Counter[int] = Counter()
        verified_votes: Counter[int] = Counter()
        answered_attempts = 0

        for a in attempts:
            ans = _as_int(a.get("answer"))
            if ans is None:
                continue
            answered_attempts += 1
            votes[ans] += 1
            py_calls = _safe_int(a.get("python_calls"), 0)
            py_errors = _safe_int(a.get("python_errors"), 0)
            if py_calls > 0 and py_errors == 0:
                verified_votes[ans] += 1

        chosen = _as_int(end.get("answer"))
        time_s = _safe_float(end.get("time_s"), 0.0)
        attempts_total = _safe_int(end.get("attempts_total"), len(attempts))

        top_vote_answer = None
        top_vote_count = 0
        if votes:
            top_vote_answer, top_vote_count = votes.most_common(1)[0]

        chosen_vote_count = votes.get(chosen, 0) if chosen is not None else 0
        chosen_verified_count = (
            verified_votes.get(chosen, 0) if chosen is not None else 0
        )
        top_vote_verified_count = (
            verified_votes.get(top_vote_answer, 0) if top_vote_answer is not None else 0
        )

        ranking = end.get("ranking")
        ranking_top_answer = None
        ranking_top_votes = 0
        ranking_top_verified = 0
        if isinstance(ranking, list) and ranking and isinstance(ranking[0], dict):
            ranking_top_answer = _as_int(ranking[0].get("answer"))
            ranking_top_votes = _safe_int(ranking[0].get("votes"), 0)
            ranking_top_verified = _safe_int(ranking[0].get("verified"), 0)

        likely_verified_filter_failure = False
        if (
            chosen is not None
            and top_vote_answer is not None
            and chosen != top_vote_answer
        ):
            # Signature failure:
            # - chosen has verified support
            # - top-vote answer has strictly more votes
            # - top-vote answer has no verified support
            # This matches "filter_to_verified_if_any=True" behavior.
            if (
                chosen_verified_count > 0
                and top_vote_count > chosen_vote_count
                and top_vote_verified_count == 0
            ):
                likely_verified_filter_failure = True

        analyses.append(
            ProblemAnalysis(
                problem_id=pid,
                chosen=chosen,
                time_s=time_s,
                attempts_total=attempts_total,
                answered_attempts=answered_attempts,
                distinct_answers=len(votes),
                top_vote_answer=top_vote_answer,
                top_vote_count=top_vote_count,
                chosen_vote_count=chosen_vote_count,
                chosen_verified_count=chosen_verified_count,
                top_vote_verified_count=top_vote_verified_count,
                ranking_top_answer=ranking_top_answer,
                ranking_top_votes=ranking_top_votes,
                ranking_top_verified=ranking_top_verified,
                likely_verified_filter_failure=likely_verified_filter_failure,
            )
        )

    analyses.sort(key=lambda x: x.problem_id)
    return analyses


def _parse_attempt(ev: dict[str, Any]) -> AttemptSummary:
    py_calls = _safe_int(ev.get("python_calls"), 0)
    py_errors = _safe_int(ev.get("python_errors"), 0)
    answer = _as_int(ev.get("answer"))
    calls_text_raw = ev.get("python_calls_text")
    outs_text_raw = ev.get("python_outputs_text")
    calls_text = calls_text_raw if isinstance(calls_text_raw, list) else []
    outs_text = outs_text_raw if isinstance(outs_text_raw, list) else []

    return AttemptSummary(
        problem_id=str(ev.get("problem_id") or ""),
        attempt=_safe_int(ev.get("attempt"), 0),
        tag=str(ev.get("tag") or ""),
        answer=answer,
        token_count=_safe_int(ev.get("token_count"), 0),
        python_calls=py_calls,
        python_errors=py_errors,
        timeout_count=_safe_int(ev.get("timeout_count"), 0),
        deadline_exceeded=bool(ev.get("deadline_exceeded", False)),
        is_verified=(py_calls > 0 and py_errors == 0),
        last_error=(
            str(ev.get("last_error")).strip() if ev.get("last_error") else None
        ),
        python_calls_text=[str(x) for x in calls_text if x is not None],
        python_outputs_text=[str(x) for x in outs_text if x is not None],
        full_reasoning_text=(
            str(ev.get("full_reasoning_text"))
            if ev.get("full_reasoning_text") is not None
            else None
        ),
    )


def build_attempt_summaries(path: Path) -> dict[str, list[AttemptSummary]]:
    """Build per-problem attempt summaries from attempt_end events."""
    _solve_end_by_pid, attempt_end_by_pid = load_event_groups(path)
    out: dict[str, list[AttemptSummary]] = {}
    for pid, rows in attempt_end_by_pid.items():
        parsed = [_parse_attempt(r) for r in rows]
        parsed.sort(key=lambda x: x.attempt)
        out[pid] = parsed
    return out


def print_attempts_global_summary(
    attempts_by_pid: dict[str, list[AttemptSummary]],
) -> None:
    all_attempts: list[AttemptSummary] = []
    for arr in attempts_by_pid.values():
        all_attempts.extend(arr)

    if not all_attempts:
        print("No attempt_end events found.")
        return

    total = len(all_attempts)
    answered = sum(1 for a in all_attempts if a.answer is not None)
    verified = sum(1 for a in all_attempts if a.is_verified)
    timed_out = sum(1 for a in all_attempts if a.timeout_count > 0)
    with_errors = sum(1 for a in all_attempts if a.python_errors > 0)
    avg_tokens = sum(a.token_count for a in all_attempts) / max(1, total)
    avg_py_calls = sum(a.python_calls for a in all_attempts) / max(1, total)

    print()
    print("Attempt-level summary")
    print(f"Problems with attempts: {len(attempts_by_pid)}")
    print(f"Total attempts: {total}")
    print(f"Answered attempts: {answered} ({100.0 * answered / total:.1f}%)")
    print(f"Verified attempts: {verified} ({100.0 * verified / total:.1f}%)")
    print(
        f"Attempts with python errors: {with_errors} ({100.0 * with_errors / total:.1f}%)"
    )
    print(f"Attempts with timeout: {timed_out} ({100.0 * timed_out / total:.1f}%)")
    print(f"Avg token_count: {avg_tokens:.1f}")
    print(f"Avg python_calls: {avg_py_calls:.1f}")


def print_attempts_for_problem(
    pid: str,
    attempts_by_pid: dict[str, list[AttemptSummary]],
    *,
    max_attempts: int = 50,
    show_python: bool = False,
    show_reasoning: bool = False,
    reasoning_chars: int = 1200,
    snippet_chars: int = 300,
) -> None:
    attempts = attempts_by_pid.get(pid, [])
    if not attempts:
        print(f"No attempt_end rows found for problem_id={pid}")
        return

    print()
    print(f"Attempts for problem_id={pid}")
    print("attempt | answer | verified | py_calls | py_err | timeout | tokens | tag")
    for a in attempts[: max(1, int(max_attempts))]:
        print(
            f"{a.attempt} | {a.answer} | {a.is_verified} | {a.python_calls} | "
            f"{a.python_errors} | {a.timeout_count} | {a.token_count} | {a.tag}"
        )
        if a.last_error:
            err = a.last_error[:snippet_chars]
            print(f"  last_error: {err}")
        if show_python:
            if a.python_calls_text:
                code = a.python_calls_text[-1][:snippet_chars]
                print(f"  last_py_code: {code}")
            if a.python_outputs_text:
                out = a.python_outputs_text[-1][:snippet_chars]
                print(f"  last_py_out: {out}")
        if show_reasoning and a.full_reasoning_text:
            reasoning = a.full_reasoning_text[: max(1, int(reasoning_chars))]
            print("  full_reasoning_preview:")
            for line in reasoning.splitlines()[:80]:
                print(f"    {line}")


def export_full_reasoning_for_problem(
    pid: str,
    attempts_by_pid: dict[str, list[AttemptSummary]],
    out_dir: Path,
) -> int:
    attempts = attempts_by_pid.get(pid, [])
    if not attempts:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    for a in attempts:
        txt = str(a.full_reasoning_text or "")
        if not txt:
            continue
        p = out_dir / f"{pid}_attempt{int(a.attempt):02d}.txt"
        p.write_text(txt, encoding="utf-8")
        exported += 1
    return exported


def print_report(
    analyses: list[ProblemAnalysis], answers: dict[str, int] | None
) -> None:
    if not analyses:
        print("No solve_end events found.")
        return

    total = len(analyses)
    avg_time = sum(a.time_s for a in analyses) / max(1, total)
    flagged = [a for a in analyses if a.likely_verified_filter_failure]

    print(f"Problems: {total}")
    print(f"Avg solve time: {avg_time:.2f}s")
    print(f"Likely verified-filter failures: {len(flagged)}")

    if answers:
        covered = 0
        correct = 0
        for a in analyses:
            if a.problem_id not in answers or a.chosen is None:
                continue
            covered += 1
            if a.chosen == answers[a.problem_id]:
                correct += 1
        if covered > 0:
            print(
                f"Accuracy on provided ground truth: {correct}/{covered} ({100.0 * correct / covered:.1f}%)"
            )
        else:
            print("Accuracy on provided ground truth: no overlapping problem_ids")

    print()
    print("Top suspicious cases")
    print("problem_id | chosen(v,vf) | top_vote(v,vf) | attempts | note")
    for a in sorted(
        analyses,
        key=lambda x: (
            not x.likely_verified_filter_failure,
            -(x.top_vote_count - x.chosen_vote_count),
            -x.attempts_total,
        ),
    )[:20]:
        note = "verified_filter_risk" if a.likely_verified_filter_failure else ""
        chosen_part = (
            f"{a.chosen}({a.chosen_vote_count},{a.chosen_verified_count})"
            if a.chosen is not None
            else "None(0,0)"
        )
        top_part = (
            f"{a.top_vote_answer}({a.top_vote_count},{a.top_vote_verified_count})"
            if a.top_vote_answer is not None
            else "None(0,0)"
        )
        print(
            f"{a.problem_id} | {chosen_part} | {top_part} | {a.attempts_total} | {note}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze AIMO-3 v2 trace JSONL.")
    ap.add_argument("trace", type=str, help="Path to trace JSONL file")
    ap.add_argument(
        "--answers",
        type=str,
        default="",
        help="Optional JSON file mapping problem_id -> true answer",
    )
    ap.add_argument(
        "--attempts-summary",
        action="store_true",
        help="Also print global attempt_end statistics.",
    )
    ap.add_argument(
        "--show-attempts",
        action="store_true",
        help="Print attempt table for one problem_id (requires --problem-id).",
    )
    ap.add_argument(
        "--problem-id",
        type=str,
        default="",
        help="Problem id to inspect attempts for.",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=50,
        help="Max attempts to print with --show-attempts (default: 50).",
    )
    ap.add_argument(
        "--show-python",
        action="store_true",
        help="With --show-attempts, include a short snippet of last python code/output.",
    )
    ap.add_argument(
        "--show-reasoning",
        action="store_true",
        help="With --show-attempts, include a preview of full prompt→end reasoning text.",
    )
    ap.add_argument(
        "--reasoning-chars",
        type=int,
        default=1200,
        help="Preview chars for --show-reasoning (default: 1200).",
    )
    ap.add_argument(
        "--dump-reasoning-dir",
        type=str,
        default="",
        help="With --show-attempts, export each attempt full reasoning to this directory.",
    )
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    answers = load_answers(Path(args.answers)) if args.answers else None
    analyses = analyze_trace(trace_path)
    print_report(analyses, answers)

    if args.attempts_summary or args.show_attempts:
        attempts_by_pid = build_attempt_summaries(trace_path)
        if args.attempts_summary:
            print_attempts_global_summary(attempts_by_pid)
        if args.show_attempts:
            if not args.problem_id:
                raise ValueError("--show-attempts requires --problem-id")
            print_attempts_for_problem(
                args.problem_id,
                attempts_by_pid,
                max_attempts=args.max_attempts,
                show_python=bool(args.show_python),
                show_reasoning=bool(args.show_reasoning),
                reasoning_chars=args.reasoning_chars,
            )
            if args.dump_reasoning_dir:
                out_dir = Path(args.dump_reasoning_dir)
                n = export_full_reasoning_for_problem(
                    args.problem_id,
                    attempts_by_pid,
                    out_dir,
                )
                print(f"Exported {n} reasoning files to {out_dir}")


if __name__ == "__main__":
    main()
