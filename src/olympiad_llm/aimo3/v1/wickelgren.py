from __future__ import annotations

"""Wickelgren-inspired problem-solving strategies.

This module provides *paraphrased* strategy checklists inspired by classical
math problem-solving heuristics (including Wickelgren-style guidance), without
reproducing any book text.

Goal: reduce prompt brittleness by giving the model a concrete, varied
"strategy card" each attempt (understand → explore → plan → execute → check).
"""

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyCard:
    name: str
    instructions: list[str]


# =============================================================================
# REWRITTEN STRATEGY CARDS: Concise, concrete, action-oriented
# Each card is a SHORT directive that biases the model toward a specific approach
# =============================================================================

GENERIC_STRATEGY_CARDS: list[StrategyCard] = [
    StrategyCard(
        name="brute_force_first",
        instructions=[
            "IMMEDIATELY write Python code to compute small cases (n=1,2,3,... or enumerate).",
            "Print results clearly. Look for a pattern in the output.",
            "Once you see the pattern, state your conjecture and verify with more cases.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="closed_form_hunt",
        instructions=[
            "Compute the first 5-10 values using Python.",
            "Search OEIS-style: does it match factorials, Catalan, Fibonacci, powers, or binomials?",
            "Test your closed-form formula on ALL computed cases before finalizing.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="modular_arithmetic",
        instructions=[
            "The answer is mod some number. Compute the base value first, then reduce.",
            "For large exponents: use pow(base, exp, mod) in Python.",
            "Watch for: Fermat's little theorem, Chinese remainder theorem, lifting the exponent.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="case_analysis",
        instructions=[
            "Split the problem into 2-3 cases based on parity, sign, or divisibility.",
            "Solve each case separately with Python verification.",
            "Combine cases carefully—don't double-count or miss edge cases.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="work_backwards",
        instructions=[
            "Start from the answer format. What structure must the answer have?",
            "Work backwards: what conditions force this structure?",
            "Use Python to check if your backwards reasoning produces valid examples.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="reduce_to_known",
        instructions=[
            "Can this reduce to: GCD/LCM? Binomial coefficient? Sum of divisors? Euler phi?",
            "Use sympy: factorint, divisors, totient, binomial, factorial.",
            "Verify the reduction is correct on small examples.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="generate_and_test",
        instructions=[
            "Write Python to generate ALL valid objects (permutations, subsets, sequences).",
            "Count or filter them according to the problem conditions.",
            "For large n, find a recurrence or closed form from small-n data.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="algebraic_manipulation",
        instructions=[
            "Use sympy to expand, factor, simplify, or solve symbolically.",
            "Don't do algebra by hand—let the computer handle it.",
            "Verify symbolic results numerically with concrete values.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999}.",
        ],
    ),
]


FE_COMBI_STRATEGY_CARDS: list[StrategyCard] = [
    StrategyCard(
        name="fe_substitution",
        instructions=[
            "Plug in special values: f(0), f(1), f(-1), f(x,x), f(x,0), f(0,y).",
            "Write Python to test if f is: constant, identity, linear ax+b, multiplicative.",
            "Find ALL solutions—don't stop at the first one that works.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
    StrategyCard(
        name="counting_small",
        instructions=[
            "Write Python to enumerate and count for small n (n=1,2,3,4,5).",
            "Store results in a list. Look for: doubling, factorial growth, polynomial pattern.",
            "Fit a formula and verify it predicts the next case correctly.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999}.",
        ],
    ),
    StrategyCard(
        name="inclusion_exclusion",
        instructions=[
            "Identify what to count and what constraints to satisfy.",
            "Apply inclusion-exclusion: count(A or B) = count(A) + count(B) - count(A and B).",
            "Verify with brute-force enumeration on small cases.",
            "Return the final answer in \\boxed{n}, where n ∈ [0, 99999].",
        ],
    ),
]


@dataclass(frozen=True)
class StrategyPack:
    name: str
    cards: list[StrategyCard]


PACKS: dict[str, StrategyPack] = {
    "generic": StrategyPack(name="generic", cards=GENERIC_STRATEGY_CARDS),
    "fe_combi": StrategyPack(name="fe_combi", cards=FE_COMBI_STRATEGY_CARDS),
}


_FE_COMBI_CUE_RE = re.compile(
    r"(functional\s+equation|find\s+all\s+functions|f\(|g\(|h\(|\bpermutation\b|\bsubset\b|\bsubsets\b|\bgraph\b|\bcombin\w+\b|\bcount\b|\barrange\b)",
    flags=re.IGNORECASE,
)


def _parse_enabled_packs(enabled: str | list[str] | None) -> list[str]:
    if enabled is None:
        return ["generic"]
    if isinstance(enabled, list):
        packs = [p.strip() for p in enabled if str(p).strip()]
    else:
        packs = [p.strip() for p in str(enabled).split(",") if p.strip()]
    packs = [p for p in packs if p in PACKS]
    return packs or ["generic"]


def detect_fe_combi(problem_text: str | None) -> bool:
    return bool(_FE_COMBI_CUE_RE.search(problem_text or ""))


def _problem_seed(problem_text: str | None) -> int:
    """Generate a stable seed from problem text for per-problem shuffling."""
    if not problem_text:
        return 0
    h = hashlib.sha1(problem_text.encode("utf-8"), usedforsecurity=False)
    return int(h.hexdigest()[:8], 16)


def select_strategy(
    attempt_index: int,
    *,
    pack: str = "generic",
    problem_text: str | None = None,
    shuffle: bool = True,
) -> StrategyCard:
    """Select a strategy card for this attempt.

    If shuffle=True (default), cards are shuffled per-problem using a deterministic
    seed from the problem text. This provides:
    - Full coverage of all cards within each problem (round-robin through shuffled order)
    - Diversity across problems (different priority order per problem)
    - Reproducibility (same problem → same shuffle)
    """
    p = PACKS.get(pack, PACKS["generic"])
    if not p.cards:
        raise RuntimeError("No strategy cards configured")

    cards = list(p.cards)

    if shuffle and problem_text:
        seed = _problem_seed(problem_text)
        rng = random.Random(seed)
        rng.shuffle(cards)

    return cards[int(attempt_index) % len(cards)]


def select_strategy_pack(
    *,
    attempt_index: int,
    problem_text: str | None,
    mode: str,
    enabled_packs: str | list[str] | None,
) -> str:
    packs = _parse_enabled_packs(enabled_packs)
    m = (mode or "").strip().lower()
    if m in {"off", "none", "0"}:
        return "generic"

    if m == "auto":
        if detect_fe_combi(problem_text) and "fe_combi" in packs:
            # Prioritize the topic pack on the first attempt, then alternate.
            # This keeps behavior general (still multi-attempt diverse), while
            # ensuring at least one targeted attempt gets run early.
            packs = ["fe_combi", "generic"] if "generic" in packs else ["fe_combi"]
            return packs[int(attempt_index) % len(packs)]

        return "generic"

    # round-robin default
    return packs[int(attempt_index) % len(packs)]


def render_strategy_card(card: StrategyCard) -> str:
    """Render a strategy card as a SHORT, actionable directive."""
    lines = [f"STRATEGY [{card.name}]:"]
    for item in card.instructions:
        lines.append(f"• {item}")
    return "\n".join(lines)


def augment_system_prompt_with_meta(
    base_prompt: str,
    *,
    attempt_index: int,
    problem_text: str | None = None,
    mode: str = "round_robin",
    enabled_packs: str | list[str] | None = None,
    shuffle_cards: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Append a strategy card and return (prompt, meta).

    Meta includes the selected pack and card name for attempt tagging.

    If shuffle_cards=True (default), cards within each pack are shuffled per-problem
    using a deterministic seed from the problem text. This ensures:
    - Full coverage of all cards within each problem
    - Different priority order across problems (diversity)
    - Reproducibility (same problem text → same shuffle)
    """

    pack = select_strategy_pack(
        attempt_index=int(attempt_index),
        problem_text=problem_text,
        mode=mode,
        enabled_packs=enabled_packs,
    )
    card = select_strategy(
        int(attempt_index),
        pack=pack,
        problem_text=problem_text,
        shuffle=shuffle_cards,
    )
    strategy_text = render_strategy_card(card)
    out = (
        strategy_text
        if not base_prompt
        else base_prompt.rstrip() + "\n\n" + strategy_text
    )
    return out, {"pack": pack, "card": card.name}


def augment_system_prompt(base_prompt: str, *, attempt_index: int) -> str:
    """Append a strategy card to the base system prompt.

    Backward-compatible wrapper: uses only the generic pack.
    """

    out, _meta = augment_system_prompt_with_meta(
        base_prompt,
        attempt_index=attempt_index,
        problem_text=None,
        mode="off",
        enabled_packs=["generic"],
    )
    return out
