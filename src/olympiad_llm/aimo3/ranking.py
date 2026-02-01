from __future__ import annotations

"""Candidate answer aggregation + ranking.

Goal: prefer answers supported by *clean tool runs* (python_calls > 0 and python_errors == 0),
because they are much less likely to be "plausible but wrong".

This module is deliberately lightweight and has no optional dependencies.
"""

import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .attempts import AttemptResult


def _env_bool(key: str, default: bool = True) -> bool:
    """Read a boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("0", "false", "no", "off"):
        return False
    if val in ("1", "true", "yes", "on"):
        return True
    return default


@dataclass(frozen=True)
class CandidateStats:
    answer: int
    votes: int
    verified: int
    tool_attempts: int
    tool_error_attempts: int
    timeout_attempts: int  # NEW: count of attempts that timed out
    tag_diversity: int
    calls: int
    errors: int
    len_sum: int
    n: int

    # Optional: confidence proxy aggregated from attempts.
    # We track a sum of (1/entropy) across attempts for this answer.
    # Higher is better (more confident on average).
    entropy_score_sum: float

    @property
    def avg_len(self) -> float:
        return self.len_sum / max(1, self.n)

    @property
    def entropy_score(self) -> float:
        return float(self.entropy_score_sum)

    def as_dict(self) -> dict[str, Any]:
        return {
            "votes": int(self.votes),
            "verified": int(self.verified),
            "tool_attempts": int(self.tool_attempts),
            "tool_error_attempts": int(self.tool_error_attempts),
            "timeout_attempts": int(self.timeout_attempts),
            "tag_diversity": int(self.tag_diversity),
            "calls": int(self.calls),
            "errors": int(self.errors),
            "avg_len": float(self.avg_len),
            "entropy_score": float(self.entropy_score),
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
    entropy_score_sum: dict[int, float] = {}
    tags: dict[int, set[str]] = {}

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
            stats[a] = {
                "votes": 0,
                "verified": 0,
                "tool_attempts": 0,
                "tool_error_attempts": 0,
                "timeout_attempts": 0,
                "calls": 0,
                "errors": 0,
                "len_sum": 0,
                "n": 0,
            }
            entropy_score_sum[a] = 0.0
            tags[a] = set()

        s = stats[a]
        s["votes"] += 1
        s["calls"] += int(ar.stats.python_calls)
        s["errors"] += int(ar.stats.python_errors)
        if int(ar.stats.python_calls) > 0:
            s["tool_attempts"] += 1
        if int(ar.stats.python_errors) > 0:
            s["tool_error_attempts"] += 1
        # Track timeout attempts for penalty.
        if getattr(ar.stats, "had_timeout", False):
            s["timeout_attempts"] += 1
        s["len_sum"] += int(ar.stats.token_count)
        s["n"] += 1
        if ar.stats.tool_verified:
            s["verified"] += 1

        # Optional: incorporate mean entropy if present.
        # Lower entropy => higher score contribution.
        try:
            ent = float(getattr(ar.stats, "mean_entropy", float("inf")))
        except Exception:  # noqa: BLE001
            ent = float("inf")
        if ent > 0.0 and ent != float("inf") and not math.isnan(ent):
            # Use a soft cap to avoid single ultra-low values dominating.
            entropy_score_sum[a] = float(entropy_score_sum.get(a, 0.0)) + (1.0 / max(ent, 1e-9))

        if getattr(ar, "tag", None):
            tags[a].add(str(ar.tag))

    out: list[CandidateStats] = []
    for a, s in stats.items():
        out.append(
            CandidateStats(
                answer=a,
                votes=int(s["votes"]),
                verified=int(s["verified"]),
                tool_attempts=int(s["tool_attempts"]),
                tool_error_attempts=int(s["tool_error_attempts"]),
                timeout_attempts=int(s["timeout_attempts"]),
                tag_diversity=len(tags.get(a, set())),
                calls=int(s["calls"]),
                errors=int(s["errors"]),
                len_sum=int(s["len_sum"]),
                n=int(s["n"]),
                entropy_score_sum=float(entropy_score_sum.get(a, 0.0)),
            )
        )
    return out


def _magnitude_bucket(x: int) -> int:
    """Return magnitude bucket: 0 for 0, else floor(log10(abs(x)))."""
    if x == 0:
        return 0
    return int(math.floor(math.log10(abs(x) + 1)))


def _detect_magnitude_outlier(
    candidates: list[CandidateStats],
) -> tuple[bool, int | None, set[int]]:
    """Detect if there's a magnitude outlier that might be correct.

    Returns (is_suspicious, dominant_bucket, outlier_answers).

    We flag as suspicious when:
    - Multiple answers cluster in one magnitude bucket (e.g., 1-20)
    - But one or more answers are in a much higher bucket (e.g., 8000+)
    - The clustered small answers could be "easy wrong" answers

    This helps avoid picking answer=15 when the true answer is 8687.
    """
    if len(candidates) < 3:
        return False, None, set()

    # Count answers by magnitude bucket
    bucket_counts: Counter[int] = Counter()
    bucket_answers: dict[int, list[int]] = {}

    for c in candidates:
        bucket = _magnitude_bucket(c.answer)
        bucket_counts[bucket] += c.votes  # Weight by votes
        if bucket not in bucket_answers:
            bucket_answers[bucket] = []
        bucket_answers[bucket].append(c.answer)

    if len(bucket_counts) < 2:
        return False, None, set()

    # Find dominant bucket (most votes)
    sorted_buckets = bucket_counts.most_common()
    dominant_bucket, dominant_votes = sorted_buckets[0]

    # Check for outliers: buckets that are 2+ orders of magnitude higher
    outlier_answers: set[int] = set()
    for bucket, answers in bucket_answers.items():
        if bucket >= dominant_bucket + 2:  # 2+ orders of magnitude higher
            outlier_answers.update(answers)

    # Flag as suspicious if:
    # - Dominant bucket has many votes (consensus on small answers)
    # - But there are outliers with much larger magnitude
    total_votes = sum(bucket_counts.values())
    if outlier_answers and dominant_votes >= total_votes * 0.5:
        return True, dominant_bucket, outlier_answers

    return False, dominant_bucket, set()


def rank_candidates(
    detailed_results: list[Any],
    *,
    filter_to_verified_if_any: bool = True,
    magnitude_aware: bool | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Rank candidate answers.

    Returns a list of (answer, stats_dict) pairs to keep compatibility with older code.

    Ranking is *verified-first*: if any candidate has at least one clean tool run,
    we (optionally) discard candidates with verified==0.

    If magnitude_aware=True and answers span wildly different magnitudes (e.g.,
    most answers are 1-20 but one is 8000+), we boost the outlier to avoid
    picking "easy wrong" small answers when the true answer is large.
    """
    # Allow disabling magnitude awareness via env var
    if magnitude_aware is None:
        magnitude_aware = _env_bool("AIMO3_MAGNITUDE_AWARE_RANKING", default=True)

    candidates = aggregate_candidates(detailed_results)
    if not candidates:
        return []

    # Detect magnitude outliers before filtering
    is_suspicious, dominant_bucket, outlier_answers = _detect_magnitude_outlier(candidates)

    # When magnitude is suspicious, don't filter to verified only
    # because the verified small answers might all be wrong
    should_filter_verified = filter_to_verified_if_any
    if magnitude_aware and is_suspicious:
        # Check if any outlier has tool attempts (even if not fully verified)
        outlier_candidates = [c for c in candidates if c.answer in outlier_answers]
        if any(c.tool_attempts > 0 for c in outlier_candidates):
            # Don't filter - give outliers a chance
            should_filter_verified = False

    if should_filter_verified and any(c.verified > 0 for c in candidates):
        candidates = [c for c in candidates if c.verified > 0] or candidates

    # Magnitude boost: if suspicious pattern detected, give outliers a significant boost
    # This helps when most attempts get small wrong answers but one gets a large answer
    def magnitude_boost(c: CandidateStats) -> tuple[int, int]:
        """Returns (verified_boost, vote_boost) for sorting."""
        if not magnitude_aware or not is_suspicious:
            return (0, 0)
        if c.answer in outlier_answers:
            # Strong boost: treat as if it had 1 extra verified + 3 extra votes
            return (1, 3)
        return (0, 0)

    # Verified-first, then votes.
    # Penalize tool errors and timeouts strongly. Tag diversity is helpful, but should
    # usually be a tie-breaker rather than dominating vote strength.
    candidates_sorted = sorted(
        candidates,
        key=lambda c: (
            int(c.verified > 0) + magnitude_boost(c)[0],  # Boost verified status
            c.verified + magnitude_boost(c)[0],
            c.votes + magnitude_boost(c)[1],  # Add vote boost
            # Confidence tie-breaker (only meaningful if logprobs/entropy was computed).
            c.entropy_score,
            -c.tool_error_attempts,
            -c.timeout_attempts,  # NEW: penalize answers from timed-out attempts
            -c.errors,
            c.tag_diversity,
            c.tool_attempts,
            c.calls,
            -c.avg_len,
        ),
        reverse=True,
    )

    return [(c.answer, c.as_dict()) for c in candidates_sorted]
