"""Compact skill/failure memory for v2 prompt augmentation.

This module implements a tiny T2-style memory layer: it retrieves a handful of
distilled strategy hints and recurring failure watch-outs based on the current
problem text. The goal is not to create a broad agent memory, but to inject a
few useful abstractions with very low token overhead.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .meta_learning import ProblemEmbedder


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "let",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "to",
    "we",
    "with",
}


@dataclass(frozen=True)
class MemorySnippet:
    kind: str
    title: str
    guidance: tuple[str, ...]
    triggers: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    source: str = "builtin"


def _tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    return [t for t in toks if len(t) >= 2 and t not in _STOPWORDS]


def _as_tuple_str(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(s.strip() for s in value.split("|") if s.strip())
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return tuple(out)
    s = str(value).strip()
    return (s,) if s else ()


def _snippet_from_obj(obj: Any) -> MemorySnippet | None:
    if isinstance(obj, MemorySnippet):
        return obj
    if not isinstance(obj, dict):
        return None

    kind = str(obj.get("kind", "") or "").strip().lower()
    title = str(obj.get("title", "") or "").strip()
    guidance = _as_tuple_str(obj.get("guidance"))
    if not kind or not title or not guidance:
        return None
    return MemorySnippet(
        kind=kind,
        title=title,
        guidance=guidance,
        triggers=_as_tuple_str(obj.get("triggers")),
        domains=_as_tuple_str(obj.get("domains")),
        source=str(obj.get("source", "file") or "file"),
    )


def _format_lines(kind: str, rows: list[tuple[MemorySnippet, float]], max_chars: int) -> list[str]:
    if not rows:
        return []

    section_title = "Skill memory" if kind == "skill" else "Failure watch-outs"
    lines = [f"[{section_title.upper().replace(' ', '_')}]"]
    for rank, (snippet, score) in enumerate(rows, start=1):
        text = " ".join(snippet.guidance).strip().replace("\n", " ")
        if max_chars > 0 and len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."
        lines.append(f"{rank}. [{score:.3f}] {snippet.title}: {text}")
    lines.append(f"[/{section_title.upper().replace(' ', '_')}]")
    return lines


def _builtin_snippets() -> list[MemorySnippet]:
    return [
        MemorySnippet(
            kind="skill",
            title="Exact modular arithmetic",
            guidance=(
                "Reduce the problem exactly modulo a useful base before any guesswork.",
                "Use valuations, CRT, and exact exponent rules instead of floats.",
            ),
            triggers=("mod", "remainder", "divisible", "congruence", "prime"),
            domains=("number_theory",),
        ),
        MemorySnippet(
            kind="skill",
            title="Counting and recurrence",
            guidance=(
                "Enumerate tiny cases first, then derive a recurrence or invariant.",
                "If a count looks structured, check inclusion-exclusion or generating functions.",
            ),
            triggers=("count", "ways", "arrangement", "subset", "permutation"),
            domains=("combinatorics",),
        ),
        MemorySnippet(
            kind="skill",
            title="Algebraic factoring",
            guidance=(
                "Factor, substitute, and use symmetry before expanding brute force.",
                "Test Vieta-style relations or polynomial identities on small cases.",
            ),
            triggers=("equation", "polynomial", "factor", "root", "identity"),
            domains=("algebra",),
        ),
        MemorySnippet(
            kind="skill",
            title="Geometry coordinates",
            guidance=(
                "Switch to coordinates or vectors when the picture is cluttered.",
                "Use power of a point, angle chasing, or barycentric coordinates only when they simplify the exact relations.",
            ),
            triggers=("triangle", "circle", "angle", "point", "line"),
            domains=("geometry",),
        ),
        MemorySnippet(
            kind="skill",
            title="Diophantine bounds",
            guidance=(
                "Look for modular obstructions and bounding arguments before brute force.",
                "Use descent or factorization when equations are integral.",
            ),
            triggers=("integer", "diophantine", "prime", "gcd", "lcm"),
            domains=("number_theory", "algebra"),
        ),
        MemorySnippet(
            kind="skill",
            title="Sequence structure",
            guidance=(
                "Check recurrence, monotonicity, and closed-form structure from a few exact values.",
                "Avoid floating approximations when the sequence is exact.",
            ),
            triggers=("sequence", "recurrence", "term", "iterate", "formula"),
            domains=("algebra", "combinatorics"),
        ),
        MemorySnippet(
            kind="failure",
            title="False consensus",
            guidance=(
                "Do not trust agreement alone; ensure at least one computed attempt supports the answer.",
                "A popular answer can still be a consistent wrong turn.",
            ),
            triggers=("consensus", "votes", "agreement", "majority"),
            domains=("combinatorics", "algebra", "number_theory"),
        ),
        MemorySnippet(
            kind="failure",
            title="Early boxed answer",
            guidance=(
                "Do not lock onto an early boxed value before the derivation is finished.",
                "Keep the last-resort scan as a fallback, not as the primary decision rule.",
            ),
            triggers=("boxed", "answer", "final", "extract"),
            domains=("algebra", "number_theory", "combinatorics"),
        ),
        MemorySnippet(
            kind="failure",
            title="Float rounding",
            guidance=(
                "Avoid floating-point shortcuts unless they are provably safe.",
                "Prefer exact integer or rational arithmetic throughout.",
            ),
            triggers=("float", "approx", "decimal", "ratio", "precision"),
            domains=("algebra", "geometry", "number_theory"),
        ),
        MemorySnippet(
            kind="failure",
            title="Tool success without math success",
            guidance=(
                "A clean tool run does not prove the mathematics is correct.",
                "Check the logical derivation, not just the absence of runtime errors.",
            ),
            triggers=("python", "tool", "code", "verify"),
            domains=("algebra", "number_theory", "combinatorics"),
        ),
        MemorySnippet(
            kind="failure",
            title="Rambling search",
            guidance=(
                "If the attempt is wandering, cut it off and restart with a cleaner invariant or exact formula.",
                "Long output is not a substitute for progress.",
            ),
            triggers=("long", "rambling", "wandering", "search"),
            domains=("combinatorics", "algebra", "geometry"),
        ),
        MemorySnippet(
            kind="failure",
            title="Pattern illusion",
            guidance=(
                "A few small cases are evidence, not proof.",
                "Always test a conjecture against an exact derivation or counterexample search.",
            ),
            triggers=("pattern", "guess", "conjecture", "example"),
            domains=("algebra", "number_theory", "combinatorics"),
        ),
    ]


class AgentMemoryRetriever:
    """Compact memory retriever for skill and failure hints."""

    def __init__(self, snippets: list[MemorySnippet]):
        self.snippets = snippets
        self._embedder = ProblemEmbedder()
        self._skill_snippets = [s for s in snippets if s.kind == "skill"]
        self._failure_snippets = [s for s in snippets if s.kind == "failure"]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "AgentMemoryRetriever":
        candidates: list[MemorySnippet] = []
        raw_rows: list[Any] = []

        if path is not None:
            p = Path(path).expanduser()
            if p.exists():
                raw = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    raw_rows = raw
                else:
                    raise ValueError("agent memory file must contain a list")

        for row in raw_rows:
            snippet = _snippet_from_obj(row)
            if snippet is not None:
                candidates.append(snippet)

        if not candidates:
            candidates = _builtin_snippets()

        return cls(candidates)

    @staticmethod
    def _score_snippet(query_tokens: set[str], query_lower: str, snippet: MemorySnippet, features: Any) -> float:
        snippet_tokens = set(_tokenize(snippet.title))
        snippet_tokens.update(_tokenize(" ".join(snippet.triggers)))
        snippet_tokens.update(_tokenize(" ".join(snippet.guidance)))

        overlap = len(query_tokens & snippet_tokens)
        phrase_boost = sum(1.25 for trig in snippet.triggers if " " in trig and trig in query_lower)

        domain_boost = 0.0
        for domain in snippet.domains:
            if domain == "number_theory":
                domain_boost += float(getattr(features, "has_number_theory", 0.0) or 0.0)
            elif domain == "combinatorics":
                domain_boost += float(getattr(features, "has_combinatorics", 0.0) or 0.0)
            elif domain == "algebra":
                domain_boost += float(getattr(features, "has_algebra", 0.0) or 0.0)
            elif domain == "geometry":
                domain_boost += float(getattr(features, "has_geometry", 0.0) or 0.0)

        kind_bonus = 0.25 if snippet.kind == "failure" else 0.0
        scale = math.sqrt(max(1, len(snippet_tokens)))
        return (float(overlap) + phrase_boost + domain_boost + kind_bonus) / scale

    def retrieve_for_problem(
        self,
        problem: str,
        skill_top_k: int = 2,
        failure_top_k: int = 2,
        min_score: float = 0.15,
        max_chars_per_item: int = 220,
    ) -> tuple[str, dict[str, Any]]:
        query = str(problem or "")
        query_lower = query.lower()
        query_tokens = set(_tokenize(query))
        features = self._embedder.embed(query)

        def _select(snippets: list[MemorySnippet], top_k: int) -> list[tuple[MemorySnippet, float]]:
            if top_k <= 0:
                return []
            scored = [
                (snippet, self._score_snippet(query_tokens, query_lower, snippet, features))
                for snippet in snippets
            ]
            scored = [item for item in scored if item[1] >= float(min_score)]
            scored.sort(key=lambda item: item[1], reverse=True)
            scored = scored[: max(1, int(top_k))]
            return scored

        selected_skills = _select(self._skill_snippets, skill_top_k)
        selected_failures = _select(self._failure_snippets, failure_top_k)

        blocks: list[str] = ["[META_AGENT_MEMORY]", "Reference only; do not treat as part of the problem."]
        if selected_skills:
            blocks.extend(_format_lines("skill", selected_skills, max_chars_per_item))
        if selected_failures:
            blocks.extend(_format_lines("failure", selected_failures, max_chars_per_item))
        blocks.append("[/META_AGENT_MEMORY]")

        selected_count = len(selected_skills) + len(selected_failures)
        metadata = {
            "backend": "builtin" if any(s.source == "builtin" for s in self.snippets) else "file",
            "results_count": selected_count,
            "skill_results_count": len(selected_skills),
            "failure_results_count": len(selected_failures),
            "query_len": len(query),
            "avg_score": (
                round(
                    sum(score for _, score in selected_skills + selected_failures)
                    / max(1, selected_count),
                    4,
                )
                if selected_count
                else 0.0
            ),
        }

        if selected_count == 0:
            return "", metadata
        return "\n".join(blocks), metadata


_AGENT_MEMORY_CACHE: dict[str, AgentMemoryRetriever] = {}


def init_agent_memory_from_cfg(cfg: Any) -> AgentMemoryRetriever | None:
    if not bool(getattr(cfg, "agent_memory_enabled", False)):
        return None

    memory_path = str(getattr(cfg, "agent_memory_path", "") or "").strip()
    key = memory_path or "builtin"
    if key in _AGENT_MEMORY_CACHE:
        return _AGENT_MEMORY_CACHE[key]

    retriever = AgentMemoryRetriever.load(memory_path or None)
    _AGENT_MEMORY_CACHE[key] = retriever
    return retriever
