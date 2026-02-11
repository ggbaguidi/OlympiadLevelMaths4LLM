"""Wickelgren-inspired problem-solving strategies.

This module provides *paraphrased* strategy checklists inspired by classical
math problem-solving heuristics (including Wickelgren-style guidance), without
reproducing any book text.

Goal: reduce prompt brittleness by giving the model a concrete, varied
"strategy card" each attempt (understand → explore → plan → execute → check).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyCard:
    """Concise, concrete, action-oriented problem-solving instructions."""

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
) -> tuple[str, dict[str, Any]]:
    """Append a strategy-card block to the developer prompt.

    Returns the augmented prompt and metadata for tracing/debug tags.
    """

    card = select_strategy(int(attempt_index))
    strategy_block = render_strategy_card(card)
    out = (
        strategy_block
        if not base_prompt
        else base_prompt.rstrip() + "\n\n" + strategy_block
    )
    return out, {"card": card.name}


def augment_developer_prompt(base_prompt: str, *, attempt_index: int) -> str:
    """Backward-compatible wrapper returning only the prompt text."""

    out, _meta = augment_developer_prompt_with_meta(
        base_prompt, attempt_index=attempt_index
    )
    return out
