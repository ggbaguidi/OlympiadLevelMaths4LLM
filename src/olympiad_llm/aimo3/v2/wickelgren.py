# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""Wickelgren-inspired prompt augmentation and lightweight retrieval.

This module provides:
- strategy-card augmentation for developer prompts
- CPU-friendly math knowledge retrieval (embedding-first, lexical fallback)

All injected retrieval/strategy blocks are explicitly marked as META context,
not part of the user problem statement.
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
    name: str
    instructions: list[str]


@dataclass(frozen=True)
class MathConcept:
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


def _format_retrieved_block(
    rows: list[tuple[MathConcept, float]], max_chars_per_item: int = 320
) -> str:
    if not rows:
        return ""

    lines = [
        "[META_RETRIEVED_MATH_KNOWLEDGE]",
        "This block is retrieved reference context, not part of the user problem statement.",
        "Use it only as optional support; prioritize the given problem constraints.",
    ]
    for rank, (c, score) in enumerate(rows, start=1):
        ctype = (c.concept_type or "concept").upper()
        title = f"{c.title}: " if c.title else ""
        content = (c.content or "").strip().replace("\n", " ")
        if max_chars_per_item > 0 and len(content) > max_chars_per_item:
            content = content[: max_chars_per_item - 3] + "..."
        lines.append(f"{rank}. [{ctype} score={score:.3f}] {title}{content}")
    lines.append("[/META_RETRIEVED_MATH_KNOWLEDGE]")
    return "\n".join(lines)


class FastMathRetriever:
    """CPU-only lexical retriever over v1 KB files (concepts.json/pkl)."""

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
            if c is None or not c.content.strip():
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
        q_tf = Counter(_tokenize(query))

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
        if top_k > 0:
            results = results[: max(1, int(top_k))]

        results = [
            RetrievalResult(concept=r.concept, score=r.score, rank=i + 1)
            for i, r in enumerate(results)
        ]
        metadata = {
            "backend": "lexical",
            "retrieval_time_ms": round((time.time() - start) * 1000.0, 2),
            "query_len": len(query or ""),
            "results_count": len(results),
            "top_k_requested": int(top_k),
            "min_score_threshold": float(min_score),
            "avg_score": (
                round(sum(r.score for r in results) / len(results), 4)
                if results
                else 0.0
            ),
        }
        return results, metadata

    def warmup(self) -> None:
        self.retrieve("warmup query", top_k=1, min_score=0.0)

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
        rows = [(r.concept, r.score) for r in results]
        return _format_retrieved_block(rows, max_chars_per_item), metadata


class EmbeddingMathRetriever:
    """Adapter around the v1 embedding retriever (CPU capable)."""

    def __init__(self, wrapped: Any):
        self._wrapped = wrapped

    @classmethod
    def load(
        cls, kb_dir: str | Path, model_path: str | None = None, cpu_only: bool = True
    ) -> "EmbeddingMathRetriever":
        from ..v1.math_retriever import MathRetriever as V1MathRetriever

        wrapped = V1MathRetriever.load(
            kb_dir=kb_dir,
            model_path=model_path,
            cpu_only=cpu_only,
        )
        return cls(wrapped)

    def warmup(self) -> None:
        with_errors = False
        try:
            _ = self._wrapped.encode_query("warmup query")
        except Exception:  # noqa: BLE001
            with_errors = True
        if with_errors:
            return

    def retrieve_for_problem(
        self,
        problem: str,
        top_k: int = 5,
        min_score: float = 0.08,
        include_examples: bool = True,
        include_definitions: bool = True,
        max_chars_per_item: int = 320,
    ) -> tuple[str, dict[str, Any]]:
        concept_types = None
        if not include_examples or not include_definitions:
            concept_types = ["theorem", "lemma", "corollary", "proposition", "axiom"]
            if include_definitions:
                concept_types.append("definition")
            if include_examples:
                concept_types.append("example")

        results, metadata = self._wrapped.retrieve(
            query=problem,
            top_k=top_k,
            concept_types=concept_types,
            min_score=min_score,
        )

        rows: list[tuple[MathConcept, float]] = []
        for r in results:
            c = _concept_from_obj(getattr(r, "concept", None))
            if c is None:
                continue
            score = float(getattr(r, "score", 0.0) or 0.0)
            rows.append((c, score))

        if metadata is None or not isinstance(metadata, dict):
            metadata = {}
        metadata = dict(metadata)
        metadata["backend"] = "embedding"
        return _format_retrieved_block(rows, max_chars_per_item), metadata


_RETRIEVER_CACHE: dict[str, Any] = {}


def init_math_retriever_from_cfg(cfg: Any) -> Any | None:
    """Load retriever from config, embedding-first with lexical fallback."""

    if not bool(getattr(cfg, "retriever_enabled", False)):
        return None

    kb_path = str(getattr(cfg, "retriever_knowledge_base_path", "") or "").strip()
    if not kb_path:
        return None

    backend = str(getattr(cfg, "retriever_backend", "auto") or "auto").strip().lower()
    if backend not in {"auto", "embedding", "lexical"}:
        backend = "auto"

    model_path = str(getattr(cfg, "retriever_model_path", "") or "").strip() or None
    cpu_only = bool(getattr(cfg, "retriever_cpu_only", True))
    warmup = bool(getattr(cfg, "retriever_warmup_on_init", True))
    max_concepts = int(getattr(cfg, "retriever_max_concepts", 0) or 0)

    if backend == "embedding":
        backends = ["embedding", "lexical"]
    elif backend == "lexical":
        backends = ["lexical"]
    else:
        backends = ["embedding", "lexical"]

    for b in backends:
        key = f"{b}|{kb_path}|{model_path}|{int(cpu_only)}|{int(max_concepts)}"
        if key in _RETRIEVER_CACHE:
            return _RETRIEVER_CACHE[key]

        try:
            if b == "embedding":
                retriever = EmbeddingMathRetriever.load(
                    kb_dir=kb_path,
                    model_path=model_path,
                    cpu_only=cpu_only,
                )
            else:
                retriever = FastMathRetriever.load(
                    kb_dir=kb_path,
                    max_concepts=max_concepts,
                )
            if warmup and hasattr(retriever, "warmup"):
                with_errors = False
                try:
                    retriever.warmup()
                except Exception:  # noqa: BLE001
                    with_errors = True
                if with_errors:
                    pass
            _RETRIEVER_CACHE[key] = retriever
            return retriever
        except Exception:  # noqa: BLE001
            continue

    return None


GENERIC_STRATEGY_CARDS: list[StrategyCard] = [
    StrategyCard(
        name="brute_force_first",
        instructions=[
            "Code tiny cases first.",
            "Print key values and look for a pattern.",
            "Make a conjecture, then test more cases.",
        ],
    ),
    StrategyCard(
        name="closed_form_hunt",
        instructions=[
            "Compute a few exact values.",
            "Check standard sequences or formulas.",
            "Verify the form on all sampled cases.",
        ],
    ),
    StrategyCard(
        name="modular_arithmetic",
        instructions=[
            "Compute exactly, then reduce mod the target.",
            "Use pow(a, b, m) for large exponents.",
            "Try Fermat, CRT, or valuations.",
        ],
    ),
    StrategyCard(
        name="case_analysis",
        instructions=[
            "Split by parity, sign, or divisibility.",
            "Solve each case and check with Python.",
            "Recombine carefully and cover edge cases.",
        ],
    ),
    StrategyCard(
        name="work_backwards",
        instructions=[
            "Guess the answer structure first.",
            "Work backward to necessary constraints.",
            "Check candidates with Python.",
        ],
    ),
    StrategyCard(
        name="reduce_to_known",
        instructions=[
            "Rewrite it in known objects or formulas.",
            "Use sympy helpers when helpful.",
            "Test the reduction on small examples.",
        ],
    ),
    StrategyCard(
        name="generate_and_test",
        instructions=[
            "Enumerate small valid objects.",
            "Filter or count by the constraints.",
            "Infer a rule from the small data.",
        ],
    ),
    StrategyCard(
        name="algebraic_manipulation",
        instructions=[
            "Use sympy to expand, factor, or solve.",
            "Prefer exact algebra over long manual steps.",
            "Check identities on small samples.",
        ],
    ),
]


def select_strategy(
    attempt_index: int,
    problem_text: str | None = None,
    used_strategies: list[str] | None = None,
    meta_learning_enabled: bool = True,
    meta_learning_experience_file: str | Path | None = None,
    meta_learning_exploration: float = 1.0,
    meta_learning_similarity_threshold: float = 0.7,
    preferred_strategy: str | None = None,
) -> tuple[StrategyCard, dict[str, Any]]:
    """Select strategy using meta-learning bandit when available.

    Returns tuple of (strategy_card, metadata) where metadata includes
    selection method and exploration info for tracing.
    """
    if not GENERIC_STRATEGY_CARDS:
        raise RuntimeError("No strategy cards configured")

    used = list(used_strategies or [])

    if preferred_strategy:
        for card in GENERIC_STRATEGY_CARDS:
            if card.name == preferred_strategy and card.name not in used:
                return card, {
                    "method": "preferred_strategy",
                    "exploration": False,
                    "cluster": "preferred",
                }

    # Try meta-learning bandit if problem text is available
    if problem_text and meta_learning_enabled:
        try:
            from .meta_learning import get_global_bandit, get_global_embedder

            embedder = get_global_embedder()
            features = embedder.embed(problem_text)

            strategy_names = [card.name for card in GENERIC_STRATEGY_CARDS]
            experience_file = (
                Path(str(meta_learning_experience_file).strip()).expanduser()
                if str(meta_learning_experience_file or "").strip()
                else None
            )
            bandit = get_global_bandit(
                strategy_names=strategy_names,
                exploration_factor=float(meta_learning_exploration),
                similarity_threshold=float(meta_learning_similarity_threshold),
                experience_file=experience_file,
            )

            strategy_name, meta = bandit.select_strategy(features, attempt_index, used)

            # Find the strategy card
            for card in GENERIC_STRATEGY_CARDS:
                if card.name == strategy_name:
                    return card, meta
        except Exception:
            # Fallback to rotation on any error
            pass

    # Fallback: rotation preferring strategies not used on this problem yet.
    for offset in range(len(GENERIC_STRATEGY_CARDS)):
        idx = (int(attempt_index) + offset) % len(GENERIC_STRATEGY_CARDS)
        candidate = GENERIC_STRATEGY_CARDS[idx]
        if candidate.name not in used:
            return candidate, {"method": "rotation", "exploration": True}

    card = GENERIC_STRATEGY_CARDS[int(attempt_index) % len(GENERIC_STRATEGY_CARDS)]
    return card, {"method": "rotation", "exploration": True}


def render_strategy_card(card: StrategyCard) -> str:
    lines = [
        "[META_STRATEGY_CARD]",
        "Method hints only; not part of the problem.",
        f"Card: {card.name}",
    ]
    for idx, item in enumerate(card.instructions, start=1):
        lines.append(f"{idx}. {item}")
    lines.append("Final answer: \\boxed{n}, with 0 <= n <= 99999.")
    lines.append("[/META_STRATEGY_CARD]")
    return "\n".join(lines)


def augment_developer_prompt_with_meta(
    base_prompt: str,
    *,
    attempt_index: int,
    problem_text: str | None = None,
    retriever: Any | None = None,
    retriever_top_k: int = 5,
    retriever_min_score: float = 0.08,
    retriever_include_examples: bool = True,
    retriever_include_definitions: bool = True,
    used_strategies: list[str] | None = None,
    meta_learning_enabled: bool = True,
    meta_learning_experience_file: str | Path | None = None,
    meta_learning_exploration: float = 1.0,
    meta_learning_similarity_threshold: float = 0.7,
    preferred_strategy: str | None = None,
) -> tuple[str, dict[str, Any]]:
    card, strategy_meta = select_strategy(
        int(attempt_index),
        problem_text,
        used_strategies,
        meta_learning_enabled=bool(meta_learning_enabled),
        meta_learning_experience_file=meta_learning_experience_file,
        meta_learning_exploration=float(meta_learning_exploration),
        meta_learning_similarity_threshold=float(meta_learning_similarity_threshold),
        preferred_strategy=preferred_strategy,
    )
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
        "strategy_selection_method": strategy_meta.get("method", "rotation"),
        "strategy_exploration": strategy_meta.get("exploration", False),
        "strategy_cluster": strategy_meta.get("cluster", "unknown"),
        "retriever_used": bool(retrieved_block),
        "retriever_backend": str(retrieval_meta.get("backend", "")),
        "retriever_results": int(retrieval_meta.get("results_count", 0) or 0),
        "retriever_avg_score": float(retrieval_meta.get("avg_score", 0.0) or 0.0),
    }
    return out, meta


def augment_developer_prompt(base_prompt: str, *, attempt_index: int) -> str:
    out, _meta = augment_developer_prompt_with_meta(
        base_prompt, attempt_index=attempt_index
    )
    return out
