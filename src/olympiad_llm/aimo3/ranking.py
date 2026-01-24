from __future__ import annotations

"""Candidate answer aggregation + ranking.

Goal: prefer answers supported by *clean tool runs* (python_calls > 0 and python_errors == 0),
because they are much less likely to be "plausible but wrong".

This module is deliberately lightweight and has no optional dependencies.
"""

from dataclasses import dataclass
from typing import Any

from .attempts import AttemptResult


@dataclass(frozen=True)
class CandidateStats:
    answer: int
    votes: int
    verified: int
    calls: int
    errors: int
    len_sum: int
    n: int

    @property
    def avg_len(self) -> float:
        return self.len_sum / max(1, self.n)

    def as_dict(self) -> dict[str, Any]:
        return {
            "votes": int(self.votes),
            "verified": int(self.verified),
            "calls": int(self.calls),
            "errors": int(self.errors),
            "avg_len": float(self.avg_len),
        }


def _coerce_attempt(record: Any) -> AttemptResult | None:
    """Accept AttemptResult or a legacy dict-shaped attempt record."""

    if isinstance(record, AttemptResult):
        return record

    if isinstance(record, dict):
        # Backward compatibility: allow dict-shaped records.
        answer = record.get("Answer")
        if answer is None:
            return None

        try:
            calls = int(record.get("Python Calls", 0) or 0)
            errors = int(record.get("Python Errors", 0) or 0)
            tok = int(record.get("Response Length", 0) or 0)
        except Exception:  # noqa: BLE001
            calls, errors, tok = 0, 0, 0

        # Attempt number isn't needed for aggregation.
        from ..attempts import AttemptStats  # local import to avoid cycles

        return AttemptResult(attempt=0, answer=answer, stats=AttemptStats(token_count=tok, python_calls=calls, python_errors=errors))

    return None


def aggregate_candidates(detailed_results: list[Any]) -> list[CandidateStats]:
    """Aggregate per-attempt results into per-answer candidate stats."""

    stats: dict[int, dict[str, int]] = {}

    for rec in detailed_results:
        ar = _coerce_attempt(rec)
        if ar is None:
            continue

        if ar.answer is None:
            continue

        if not isinstance(ar.answer, int):
            # AIMO expects an int final answer; ignore others in voting.
            continue

        a = int(ar.answer)
        if a not in stats:
            stats[a] = {"votes": 0, "verified": 0, "calls": 0, "errors": 0, "len_sum": 0, "n": 0}

        s = stats[a]
        s["votes"] += 1
        s["calls"] += int(ar.stats.python_calls)
        s["errors"] += int(ar.stats.python_errors)
        s["len_sum"] += int(ar.stats.token_count)
        s["n"] += 1
        if ar.stats.tool_verified:
            s["verified"] += 1

    out: list[CandidateStats] = []
    for a, s in stats.items():
        out.append(
            CandidateStats(
                answer=a,
                votes=int(s["votes"]),
                verified=int(s["verified"]),
                calls=int(s["calls"]),
                errors=int(s["errors"]),
                len_sum=int(s["len_sum"]),
                n=int(s["n"]),
            )
        )
    return out


def rank_candidates(
    detailed_results: list[Any],
    *,
    filter_to_verified_if_any: bool = True,
) -> list[tuple[int, dict[str, Any]]]:
    """Rank candidate answers.

    Returns a list of (answer, stats_dict) pairs to keep compatibility with older code.

    Ranking is *verified-first*: if any candidate has at least one clean tool run,
    we (optionally) discard candidates with verified==0.
    """

    candidates = aggregate_candidates(detailed_results)
    if not candidates:
        return []

    if filter_to_verified_if_any and any(c.verified > 0 for c in candidates):
        candidates = [c for c in candidates if c.verified > 0] or candidates

    # Verified-first, then votes.
    # Penalize errors strongly; prefer candidates that used Python (calls) and have shorter completions on average.
    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            int(c.verified > 0),
            c.verified,
            c.votes,
            -c.errors,
            c.calls,
            -c.avg_len,
        ),
        reverse=True,
    )

    return [(c.answer, c.as_dict()) for c in candidates_sorted]
