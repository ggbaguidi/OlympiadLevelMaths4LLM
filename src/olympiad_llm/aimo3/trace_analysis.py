from __future__ import annotations

"""Analyze AIMO-3 JSONL traces.

Usage:
    python -m olympiad_llm.aimo3.trace_analysis aimo3_trace.jsonl

Optionally provide a JSON mapping of problem_id -> true_answer to estimate accuracy:
    python -m olympiad_llm.aimo3.trace_analysis aimo3_trace.jsonl --answers answers.json

This module is dependency-free (stdlib only) so it can run in Kaggle.
"""

import argparse
import csv
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class ProblemSummary:
    problem_id: str
    status: str
    chosen: int
    elapsed_s: float

    budget_s: float | None = None
    attempt_deadline_in_s: float | None = None
    overall_deadline_in_s: float | None = None

    n_attempts: int = 0
    n_valid_attempts: int = 0
    n_verified_attempts: int = 0
    python_calls: int = 0
    python_errors: int = 0

    top_votes: int | None = None
    top_verified: int | None = None
    top_tag_diversity: int | None = None
    second_stage_ran: bool = False
    second_stage_choice: int | None = None

    # Derived: risk flag to prioritize inspection
    risk_score: float = 0.0


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:  # noqa: BLE001
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return int(default)


def _parse_pack_card(tag: str) -> tuple[str, str]:
    """Parse pack and card from attempt tag like 'standard|pack=generic|card=brute_force_first'."""
    pack = "none"
    card = "none"
    if not tag:
        return pack, card
    for part in tag.split("|"):
        if part.startswith("pack="):
            pack = part[5:]
        elif part.startswith("card="):
            card = part[5:]
    return pack, card


@dataclass
class PackCardStats:
    """Statistics for a pack or card."""
    name: str
    attempts: int = 0
    valid: int = 0  # produced a valid answer
    verified: int = 0  # python_calls>0 and python_errors==0
    errors: int = 0  # had tool errors
    chosen: int = 0  # was the winning answer
    avg_tokens: float = 0.0


def load_answers_json(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("answers JSON must be an object mapping problem_id -> int")
    out: dict[str, int] = {}
    for k, v in data.items():
        if v is None:
            continue
        out[str(k)] = int(v)
    return out


def summarize_trace(
    events: Iterable[dict[str, Any]],
    pack_stats: dict[str, PackCardStats] | None = None,
    card_stats: dict[str, PackCardStats] | None = None,
) -> list[ProblemSummary]:
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    attempt_ends: list[dict[str, Any]] = []  # collect attempt_end events for pack/card stats

    for ev in events:
        pid = str(ev.get("problem_id") or "")
        if not pid:
            continue
        et = ev.get("event")
        if et is None:
            et = ev.get("event_type")
        if et is None:
            et = ev.get("type")

        if et == "solve_start":
            starts[pid] = ev
        elif et == "solve_end":
            ends[pid] = ev
        elif et == "attempt_end":
            attempt_ends.append(ev)

    summaries: list[ProblemSummary] = []
    for pid, end in ends.items():
        start = starts.get(pid, {})

        status = str(end.get("status") or "")
        chosen = _safe_int(end.get("chosen"), 0)
        elapsed = _safe_float(end.get("elapsed_s"), 0.0)

        attempts = end.get("attempts") or []
        if not isinstance(attempts, list):
            attempts = []

        n_attempts = len(attempts)
        n_valid = 0
        n_verified = 0
        calls = 0
        errors = 0
        for a in attempts:
            if not isinstance(a, dict):
                continue
            ans = a.get("answer")
            if isinstance(ans, int):
                n_valid += 1
            pc = _safe_int(a.get("python_calls"), 0)
            pe = _safe_int(a.get("python_errors"), 0)
            calls += pc
            errors += pe
            if pc > 0 and pe == 0:
                n_verified += 1

        decision = end.get("decision") or {}
        ranked = (decision.get("ranked") if isinstance(decision, dict) else None) or []
        top_votes = None
        top_verified = None
        top_tag_diversity = None
        if isinstance(ranked, list) and ranked:
            r0 = ranked[0] if isinstance(ranked[0], dict) else {}
            top_votes = _safe_int(r0.get("votes"), 0)
            top_verified = _safe_int(r0.get("verified"), 0)
            top_tag_diversity = _safe_int(r0.get("tag_diversity"), 0)

        second_stage = (decision.get("second_stage") if isinstance(decision, dict) else None) or None
        second_stage_ran = isinstance(second_stage, dict)
        second_stage_choice = None
        if second_stage_ran:
            second_stage_choice = second_stage.get("choice")
            if second_stage_choice is not None:
                second_stage_choice = _safe_int(second_stage_choice, 0)

        budget_s = start.get("budget_s")
        attempt_deadline_in_s = start.get("attempt_deadline_in_s")
        overall_deadline_in_s = start.get("overall_deadline_in_s")

        ps = ProblemSummary(
            problem_id=pid,
            status=status,
            chosen=chosen,
            elapsed_s=elapsed,
            budget_s=_safe_float(budget_s, 0.0) if budget_s is not None else None,
            attempt_deadline_in_s=_safe_float(attempt_deadline_in_s, 0.0) if attempt_deadline_in_s is not None else None,
            overall_deadline_in_s=_safe_float(overall_deadline_in_s, 0.0) if overall_deadline_in_s is not None else None,
            n_attempts=n_attempts,
            n_valid_attempts=n_valid,
            n_verified_attempts=n_verified,
            python_calls=calls,
            python_errors=errors,
            top_votes=top_votes,
            top_verified=top_verified,
            top_tag_diversity=top_tag_diversity,
            second_stage_ran=second_stage_ran,
            second_stage_choice=second_stage_choice,
        )

        # Risk scoring: higher means "more likely wrong" (heuristic).
        # - no verified support is risky
        # - tool errors are risky
        # - low tag diversity is mildly risky
        risk = 0.0
        if ps.status != "ok":
            risk += 5.0
        if (ps.top_verified or 0) <= 0:
            risk += 2.0
        # If we attempted second-stage verification but couldn't select a winner,
        # that's a strong signal the run ended in ambiguity.
        if ps.second_stage_ran and ps.second_stage_choice is None:
            risk += 1.0
        if ps.python_errors > 0:
            risk += min(3.0, ps.python_errors / 5.0)
        if (ps.top_tag_diversity or 0) <= 1:
            risk += 0.5
        ps.risk_score = float(risk)

        summaries.append(ps)

    # Collect pack/card statistics from attempt_end events
    if pack_stats is not None or card_stats is not None:
        # Build a map of (problem_id, chosen_answer) for determining which attempts were chosen
        chosen_map: dict[str, int] = {}
        for pid, end in ends.items():
            chosen_map[pid] = _safe_int(end.get("chosen"), -999999)

        for ae in attempt_ends:
            tag = str(ae.get("tag") or "")
            pack, card = _parse_pack_card(tag)
            answer = ae.get("answer")
            pc = _safe_int(ae.get("python_calls"), 0)
            pe = _safe_int(ae.get("python_errors"), 0)
            tokens = _safe_int(ae.get("token_count"), 0)
            pid = str(ae.get("problem_id") or "")

            is_valid = isinstance(answer, int)
            is_verified = pc > 0 and pe == 0
            has_errors = pe > 0
            is_chosen = is_valid and pid in chosen_map and answer == chosen_map[pid]

            for stats_dict, key in [(pack_stats, pack), (card_stats, card)]:
                if stats_dict is None:
                    continue
                if key not in stats_dict:
                    stats_dict[key] = PackCardStats(name=key)
                s = stats_dict[key]
                s.attempts += 1
                if is_valid:
                    s.valid += 1
                if is_verified:
                    s.verified += 1
                if has_errors:
                    s.errors += 1
                if is_chosen:
                    s.chosen += 1
                # Running average for tokens
                s.avg_tokens = s.avg_tokens + (tokens - s.avg_tokens) / s.attempts

    # Stable order: worst first.
    summaries.sort(key=lambda s: (s.risk_score, s.elapsed_s), reverse=True)
    return summaries


def write_csv(path: Path, summaries: list[ProblemSummary], *, with_correctness: bool = False) -> None:
    fields = [
        "problem_id",
        "status",
        "chosen",
        "elapsed_s",
        "budget_s",
        "attempt_deadline_in_s",
        "overall_deadline_in_s",
        "n_attempts",
        "n_valid_attempts",
        "n_verified_attempts",
        "python_calls",
        "python_errors",
        "top_votes",
        "top_verified",
        "top_tag_diversity",
        "second_stage_ran",
        "second_stage_choice",
        "risk_score",
    ]
    if with_correctness:
        fields += ["true_answer", "is_correct"]

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in summaries:
            row = {
                "problem_id": s.problem_id,
                "status": s.status,
                "chosen": s.chosen,
                "elapsed_s": s.elapsed_s,
                "budget_s": s.budget_s,
                "attempt_deadline_in_s": s.attempt_deadline_in_s,
                "overall_deadline_in_s": s.overall_deadline_in_s,
                "n_attempts": s.n_attempts,
                "n_valid_attempts": s.n_valid_attempts,
                "n_verified_attempts": s.n_verified_attempts,
                "python_calls": s.python_calls,
                "python_errors": s.python_errors,
                "top_votes": s.top_votes,
                "top_verified": s.top_verified,
                "top_tag_diversity": s.top_tag_diversity,
                "second_stage_ran": bool(s.second_stage_ran),
                "second_stage_choice": s.second_stage_choice,
                "risk_score": s.risk_score,
            }
            if with_correctness:
                row["true_answer"] = getattr(s, "true_answer", None)
                row["is_correct"] = getattr(s, "is_correct", None)
            w.writerow(row)


def print_report(
    summaries: list[ProblemSummary],
    *,
    answers: dict[str, int] | None = None,
    top_n: int = 15,
    pack_stats: dict[str, PackCardStats] | None = None,
    card_stats: dict[str, PackCardStats] | None = None,
) -> int:
    total = len(summaries)
    if total == 0:
        print("No solve_end events found.")
        return 1

    status_counts: dict[str, int] = {}
    for s in summaries:
        status_counts[s.status] = status_counts.get(s.status, 0) + 1

    elapsed = [s.elapsed_s for s in summaries if s.elapsed_s > 0]
    p50 = statistics.median(elapsed) if elapsed else 0.0
    p90 = statistics.quantiles(elapsed, n=10)[8] if len(elapsed) >= 10 else (max(elapsed) if elapsed else 0.0)

    print("=== Trace summary ===")
    print(f"Problems: {total}")
    print("Status counts:")
    for k in sorted(status_counts.keys()):
        print(f"  {k}: {status_counts[k]}")
    print(f"Elapsed: median={p50:.1f}s p90~={p90:.1f}s")

    correct = None
    if answers is not None:
        n_known = 0
        n_correct = 0
        for s in summaries:
            if s.problem_id in answers:
                n_known += 1
                if int(s.chosen) == int(answers[s.problem_id]):
                    n_correct += 1
        correct = (n_correct, n_known)
        if n_known:
            print(f"Known-answer accuracy: {n_correct}/{n_known} = {n_correct/n_known:.3f}")

    # Print pack statistics
    if pack_stats:
        print("\n=== Strategy pack statistics ===")
        print(f"{'Pack':<15} {'Attempts':>8} {'Valid':>6} {'Verified':>8} {'Errors':>6} {'Chosen':>6} {'AvgTok':>7}")
        for name in sorted(pack_stats.keys()):
            s = pack_stats[name]
            print(f"{name:<15} {s.attempts:>8} {s.valid:>6} {s.verified:>8} {s.errors:>6} {s.chosen:>6} {s.avg_tokens:>7.0f}")

    # Print card statistics
    if card_stats:
        print("\n=== Strategy card statistics ===")
        print(f"{'Card':<25} {'Attempts':>8} {'Valid':>6} {'Verified':>8} {'Errors':>6} {'Chosen':>6} {'AvgTok':>7}")
        for name in sorted(card_stats.keys()):
            s = card_stats[name]
            print(f"{name:<25} {s.attempts:>8} {s.valid:>6} {s.verified:>8} {s.errors:>6} {s.chosen:>6} {s.avg_tokens:>7.0f}")

    print("\n=== Highest-risk problems (inspect first) ===")
    for s in summaries[: max(1, int(top_n))]:
        extra = ""
        if answers is not None and s.problem_id in answers:
            extra = " (correct)" if s.chosen == answers[s.problem_id] else f" (wrong; true={answers[s.problem_id]})"
        print(
            f"{s.problem_id} status={s.status} chosen={s.chosen} verified={s.top_verified} "
            f"tool_errs={s.python_errors} tag_div={s.top_tag_diversity} elapsed={s.elapsed_s:.1f}s risk={s.risk_score:.2f}{extra}"
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=str, help="Path to aimo3_trace.jsonl")
    ap.add_argument("--answers", type=str, default="", help="Optional JSON mapping problem_id -> true answer")
    ap.add_argument("--csv", type=str, default="", help="Optional output CSV path")
    ap.add_argument("--top", type=int, default=15, help="How many risky problems to print")
    args = ap.parse_args(argv)

    trace_path = Path(args.trace)
    answers = load_answers_json(Path(args.answers)) if args.answers else None

    # Materialize events once so we can compute schema/coverage diagnostics.
    events = list(_iter_jsonl(trace_path))
    # Schema + completeness diagnostics (helpful when traces come from different versions).
    event_counts: dict[str, int] = {}
    starts: set[str] = set()
    ends: set[str] = set()
    for ev in events:
        et = ev.get("event")
        if et is None:
            et = ev.get("event_type")
        if et is None:
            et = ev.get("type")
        et = str(et) if et is not None else "?"
        event_counts[et] = event_counts.get(et, 0) + 1

        pid = ev.get("problem_id")
        if pid:
            if et == "solve_start":
                starts.add(str(pid))
            elif et == "solve_end":
                ends.add(str(pid))

    if event_counts:
        print("=== Trace file diagnostics ===")
        print(f"Events: {len(events)}")
        for k in sorted(event_counts.keys()):
            print(f"  {k}: {event_counts[k]}")
        missing_end = sorted(starts - ends)
        missing_start = sorted(ends - starts)
        if missing_end or missing_start:
            print("Incomplete pairs:")
            if missing_end:
                print(f"  solve_start without solve_end: {len(missing_end)}")
            if missing_start:
                print(f"  solve_end without solve_start: {len(missing_start)}")
        print("")

    # Collect pack/card statistics
    pack_stats: dict[str, PackCardStats] = {}
    card_stats: dict[str, PackCardStats] = {}
    summaries = summarize_trace(events, pack_stats=pack_stats, card_stats=card_stats)

    # Attach correctness if answers provided.
    if answers is not None:
        for s in summaries:
            if s.problem_id in answers:
                setattr(s, "true_answer", int(answers[s.problem_id]))
                setattr(s, "is_correct", bool(int(s.chosen) == int(answers[s.problem_id])))

    if args.csv:
        write_csv(Path(args.csv), summaries, with_correctness=answers is not None)

    return print_report(summaries, answers=answers, top_n=args.top, pack_stats=pack_stats, card_stats=card_stats)


if __name__ == "__main__":
    raise SystemExit(main())
