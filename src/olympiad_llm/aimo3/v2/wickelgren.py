# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""Wickelgren-inspired problem-solving strategies.

This module provides *paraphrased* strategy checklists inspired by classical
math problem-solving heuristics (including Wickelgren-style guidance), without
reproducing any book text.

Goal: reduce prompt brittleness by giving the model a concrete, varied
"strategy card" each attempt (understand → explore → plan → execute → check).
"""

from __future__ import annotations

import json
import math
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StrategyCard:
    """Concise, concrete, action-oriented problem-solving instructions."""

    name: str
    instructions: list[str]


@dataclass(frozen=True)
class MathConcept:
    """Compact concept record loaded from the v1 knowledge-base format."""

    concept_type: str
    title: str | None
    content: str
    source_book: str
    chapter: str | None = None
    page: int | None = None

    def to_text(self) -> str:
        parts = []
        if self.concept_type:
            parts.append(f"[{self.concept_type}]")
        if self.title:
            parts.append(self.title)
        if self.chapter:
            parts.append(self.chapter)
        if self.content:
            parts.append(self.content)
        return " ".join(parts)


@dataclass(frozen=True)
class RetrievalResult:
    """Single retriever hit."""

    concept: MathConcept
    score: float
    rank: int


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


def _tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall((text or "").lower())
    return [t for t in toks if len(t) >= 2 and t not in _STOPWORDS]


def _concept_from_obj(obj: Any) -> MathConcept | None:
    """Best-effort conversion from JSON/pickle concept rows."""

    if isinstance(obj, MathConcept):
        return obj

    if isinstance(obj, dict):
        return MathConcept(
            concept_type=str(obj.get("concept_type", "") or ""),
            title=(str(obj.get("title")) if obj.get("title") is not None else None),
            content=str(obj.get("content", "") or ""),
            source_book=str(obj.get("source_book", "") or ""),
            chapter=(
                str(obj.get("chapter")) if obj.get("chapter") is not None else None
            ),
            page=(int(obj.get("page")) if obj.get("page") is not None else None),
        )

    # Pickled v1 concepts may be dataclass instances with the same attributes.
    concept_type = getattr(obj, "concept_type", None)
    content = getattr(obj, "content", None)
    if concept_type is None and content is None:
        return None

    title = getattr(obj, "title", None)
    source_book = getattr(obj, "source_book", "")
    chapter = getattr(obj, "chapter", None)
    page = getattr(obj, "page", None)
    try:
        page_int = int(page) if page is not None else None
    except Exception:  # noqa: BLE001
        page_int = None

    return MathConcept(
        concept_type=str(concept_type or ""),
        title=(str(title) if title is not None else None),
        content=str(content or ""),
        source_book=str(source_book or ""),
        chapter=(str(chapter) if chapter is not None else None),
        page=page_int,
    )


class FastMathRetriever:
    """CPU-only lexical retriever for the v1 knowledge-base format.

    Design goals:
    - no GPU use
    - no transformer loading at runtime
    - fast enough for per-problem retrieval
    - compatible with existing v1 KB files (concepts.json / concepts.pkl)
    """

    def __init__(self, concepts: list[MathConcept]):
        self.concepts = concepts
        self._idf: dict[str, float] = {}
        self._index: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self._doc_norms: list[float] = []
        self._build_index()

    @classmethod
    def load(cls, kb_dir: str | Path, max_concepts: int = 0) -> "FastMathRetriever":
        kb = Path(kb_dir)
        concepts_path_json = kb / "concepts.json"
        concepts_path_pkl = kb / "concepts.pkl"

        raw_rows: list[Any]
        if concepts_path_json.exists():
            raw_rows = json.loads(concepts_path_json.read_text(encoding="utf-8"))
            if not isinstance(raw_rows, list):
                raise ValueError("concepts.json must contain a list")
        elif concepts_path_pkl.exists():
            with open(concepts_path_pkl, "rb") as f:
                raw_rows = pickle.load(f)
            if not isinstance(raw_rows, list):
                raise ValueError("concepts.pkl must contain a list")
        else:
            raise FileNotFoundError(
                f"No concepts.json or concepts.pkl found in knowledge base: {kb}"
            )

        concepts: list[MathConcept] = []
        for row in raw_rows:
            c = _concept_from_obj(row)
            if c is None:
                continue
            if not c.content.strip():
                continue
            concepts.append(c)

        if max_concepts > 0:
            concepts = concepts[: max(1, int(max_concepts))]

        if not concepts:
            raise ValueError("Knowledge base has no valid concepts")

        return cls(concepts)

    def _build_index(self) -> None:
        n = len(self.concepts)
        df: Counter[str] = Counter()
        doc_tfs: list[Counter[str]] = []

        for concept in self.concepts:
            tf = Counter(_tokenize(concept.to_text()))
            doc_tfs.append(tf)
            for tok in tf:
                df[tok] += 1

        self._idf = {
            tok: math.log((n + 1.0) / (freq + 1.0)) + 1.0 for tok, freq in df.items()
        }

        self._doc_norms = [1.0] * n
        for idx, tf in enumerate(doc_tfs):
            norm_sq = 0.0
            for tok, cnt in tf.items():
                idf = self._idf.get(tok, 0.0)
                w = (1.0 + math.log(float(cnt))) * idf
                norm_sq += w * w
                self._index[tok].append((idx, 1.0 + math.log(float(cnt))))
            self._doc_norms[idx] = math.sqrt(norm_sq) if norm_sq > 0 else 1.0

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.08,
        include_examples: bool = True,
        include_definitions: bool = True,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        start = time.time()
        query_toks = _tokenize(query)
        q_tf = Counter(query_toks)

        # Query weights in the same TF-IDF space.
        q_weights: dict[str, float] = {}
        q_norm_sq = 0.0
        for tok, cnt in q_tf.items():
            idf = self._idf.get(tok)
            if idf is None:
                continue
            w = (1.0 + math.log(float(cnt))) * idf
            q_weights[tok] = w
            q_norm_sq += w * w
        q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

        raw_scores: dict[int, float] = defaultdict(float)
        for tok, qw in q_weights.items():
            idf = self._idf.get(tok, 0.0)
            for doc_idx, doc_tf_weight in self._index.get(tok, []):
                raw_scores[doc_idx] += qw * (doc_tf_weight * idf)

        results: list[RetrievalResult] = []
        for idx, dot in raw_scores.items():
            c = self.concepts[idx]
            ctype = (c.concept_type or "").strip().lower()
            if not include_examples and ctype == "example":
                continue
            if not include_definitions and ctype == "definition":
                continue

            score = dot / (q_norm * max(self._doc_norms[idx], 1e-9))
            if score < float(min_score):
                continue
            results.append(RetrievalResult(concept=c, score=float(score), rank=0))

        results.sort(key=lambda r: r.score, reverse=True)
        top_k = max(0, int(top_k))
        if top_k > 0:
            results = results[:top_k]

        # Populate ranks after sorting/slicing.
        results = [
            RetrievalResult(concept=r.concept, score=r.score, rank=i + 1)
            for i, r in enumerate(results)
        ]

        metadata = {
            "retrieval_time_ms": round((time.time() - start) * 1000.0, 2),
            "query_len": len(query or ""),
            "results_count": len(results),
            "top_k_requested": int(top_k),
            "min_score_threshold": float(min_score),
            "include_examples": bool(include_examples),
            "include_definitions": bool(include_definitions),
            "avg_score": (
                round(sum(r.score for r in results) / len(results), 4)
                if results
                else 0.0
            ),
        }
        return results, metadata

    def retrieve_for_problem(
        self,
        problem: str,
        top_k: int = 5,
        min_score: float = 0.08,
        include_examples: bool = True,
        include_definitions: bool = True,
        max_chars_per_item: int = 320,
    ) -> tuple[str, dict[str, Any]]:
        results, metadata = self.retrieve(
            query=problem,
            top_k=top_k,
            min_score=min_score,
            include_examples=include_examples,
            include_definitions=include_definitions,
        )

        if not results:
            return "", metadata

        lines = [
            "[META_RETRIEVED_MATH_KNOWLEDGE]",
            "This block is retrieved reference context, not part of the user problem statement.",
            "Use it only as optional support; prioritize the given problem constraints.",
        ]
        for r in results:
            c = r.concept
            ctype = (c.concept_type or "concept").upper()
            title = f"{c.title}: " if c.title else ""
            content = (c.content or "").strip().replace("\n", " ")
            if max_chars_per_item > 0 and len(content) > max_chars_per_item:
                content = content[: max_chars_per_item - 3] + "..."
            lines.append(f"{r.rank}. [{ctype} score={r.score:.3f}] {title}{content}")
        lines.append("[/META_RETRIEVED_MATH_KNOWLEDGE]")
        return "\n".join(lines), metadata


# Per-process retriever cache keyed by KB path.
_RETRIEVER_CACHE: dict[str, FastMathRetriever] = {}


def init_math_retriever_from_cfg(cfg: Any) -> FastMathRetriever | None:
    """Initialize/load a CPU retriever from config (compatible env names with v1)."""

    if not bool(getattr(cfg, "retriever_enabled", False)):
        return None

    kb_path = str(getattr(cfg, "retriever_knowledge_base_path", "") or "").strip()
    if not kb_path:
        return None

    if kb_path in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[kb_path]

    max_concepts = 0
    with_errors = False
    try:
        # Optional knob for huge KBs; not required in config.
        max_concepts = int(getattr(cfg, "retriever_max_concepts", 0) or 0)
    except Exception:  # noqa: BLE001
        max_concepts = 0

    try:
        retriever = FastMathRetriever.load(kb_path, max_concepts=max_concepts)
        if bool(getattr(cfg, "retriever_warmup_on_init", True)):
            # Lightweight warmup.
            retriever.retrieve(
                "warmup query",
                top_k=1,
                min_score=0.0,
                include_examples=True,
                include_definitions=True,
            )
        _RETRIEVER_CACHE[kb_path] = retriever
        return retriever
    except Exception:  # noqa: BLE001
        with_errors = True

    if with_errors:
        return None
    return None


# =============================================================================
# REWRITTEN STRATEGY CARDS: Concise, concrete, action-oriented
# Each card is a SHORT directive that biases the model toward a specific approach
# =============================================================================

GENERIC_STRATEGY_CARDS: list[StrategyCard] = [
    StrategyCard(
        name="brute_force_first",
        instructions=[
            "Start by writing Python code for small cases (for example n = 1, 2, 3, ...).",
            "Print intermediate values clearly and look for a pattern.",
            "State a conjecture from the pattern, then test it on additional cases.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
    StrategyCard(
        name="closed_form_hunt",
        instructions=[
            "Compute the first 5 to 10 values using Python.",
            "Check whether values match known sequences (factorial, Catalan, Fibonacci, powers, binomials).",
            "Validate any closed form against every computed case before concluding.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
    StrategyCard(
        name="modular_arithmetic",
        instructions=[
            "Compute the core expression first, then reduce modulo the target.",
            "For large exponents, use pow(base, exp, mod) in Python.",
            "Check common tools: Fermat, CRT, and valuation-based exponent lifting.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
    StrategyCard(
        name="case_analysis",
        instructions=[
            "Split into a small number of cases (parity, sign, or divisibility).",
            "Solve each case independently and verify with Python.",
            "Merge case results carefully and handle edge cases explicitly.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
    StrategyCard(
        name="work_backwards",
        instructions=[
            "Start from the expected answer structure and infer required constraints.",
            "Work backward from those constraints to necessary conditions.",
            "Use Python checks to validate that backward reasoning produces valid instances.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
    StrategyCard(
        name="reduce_to_known",
        instructions=[
            "Try reducing the task to known objects (gcd/lcm, binomial, divisor sum, Euler phi).",
            "Use sympy helpers such as factorint, divisors, totient, binomial, factorial.",
            "Validate the reduction on small examples before relying on it.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
    StrategyCard(
        name="generate_and_test",
        instructions=[
            "Generate all valid objects for small sizes (permutations, subsets, sequences).",
            "Filter and count according to the stated constraints.",
            "For larger n, infer a recurrence or closed form from the small-size data.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
    StrategyCard(
        name="algebraic_manipulation",
        instructions=[
            "Use sympy for expansion, factoring, simplification, and symbolic solving.",
            "Prefer computer algebra over manual symbolic manipulation.",
            "Numerically verify symbolic identities on concrete samples.",
            "End with a final answer in \\boxed{n}, where n is in [0, 99999].",
        ],
    ),
]


def select_strategy(attempt_index: int) -> StrategyCard:
    """Select a strategy card for the given attempt index."""

    if not GENERIC_STRATEGY_CARDS:
        raise RuntimeError("No strategy cards configured")
    return GENERIC_STRATEGY_CARDS[int(attempt_index) % len(GENERIC_STRATEGY_CARDS)]


def render_strategy_card(card: StrategyCard) -> str:
    """Render a strategy card as an explicit meta-instruction block.

    The block is wrapped in markers so it is less likely to be interpreted as
    part of the user's math problem statement.
    """

    lines = [
        "[META_STRATEGY_CARD]",
        "This block is solver guidance, not part of the user problem statement.",
        "Use it as a method checklist only.",
        f"Card: {card.name}",
    ]
    for idx, item in enumerate(card.instructions, start=1):
        lines.append(f"{idx}. {item}")
    lines.append("[/META_STRATEGY_CARD]")
    return "\n".join(lines)


def augment_developer_prompt_with_meta(
    base_prompt: str,
    *,
    attempt_index: int,
    problem_text: str | None = None,
    retriever: FastMathRetriever | None = None,
    retriever_top_k: int = 5,
    retriever_min_score: float = 0.08,
    retriever_include_examples: bool = True,
    retriever_include_definitions: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Append a strategy-card block to the developer prompt.

    Returns the augmented prompt and metadata for tracing/debug tags.
    """

    card = select_strategy(int(attempt_index))
    strategy_block = render_strategy_card(card)
    retrieved_block = ""
    retrieval_meta: dict[str, Any] = {}

    if retriever is not None and (problem_text or "").strip():
        with_errors = False
        try:
            retrieved_block, retrieval_meta = retriever.retrieve_for_problem(
                problem=str(problem_text or ""),
                top_k=max(1, int(retriever_top_k)),
                min_score=float(retriever_min_score),
                include_examples=bool(retriever_include_examples),
                include_definitions=bool(retriever_include_definitions),
            )
        except Exception:  # noqa: BLE001
            with_errors = True
        if with_errors:
            retrieved_block = ""
            retrieval_meta = {}

    blocks = [b for b in [strategy_block, retrieved_block] if b]
    extra = "\n\n".join(blocks)
    out = extra if not base_prompt else base_prompt.rstrip() + "\n\n" + extra
    meta = {
        "card": card.name,
        "retriever_used": bool(retrieved_block),
        "retriever_results": int(retrieval_meta.get("results_count", 0) or 0),
        "retriever_avg_score": float(retrieval_meta.get("avg_score", 0.0) or 0.0),
    }
    return out, meta


def augment_developer_prompt(base_prompt: str, *, attempt_index: int) -> str:
    """Backward-compatible wrapper returning only the prompt text."""

    out, _meta = augment_developer_prompt_with_meta(
        base_prompt, attempt_index=attempt_index
    )
    return out
