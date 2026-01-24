from __future__ import annotations

"""Wickelgren-inspired problem-solving strategies.

This module provides *paraphrased* strategy checklists inspired by classical
math problem-solving heuristics (including Wickelgren-style guidance), without
reproducing any book text.

Goal: reduce prompt brittleness by giving the model a concrete, varied
"strategy card" each attempt (understand → explore → plan → execute → check).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyCard:
    name: str
    instructions: list[str]


STRATEGY_CARDS: list[StrategyCard] = [
    StrategyCard(
        name="Understand & restate",
        instructions=[
            "Restate the problem in your own words and define every symbol.",
            "State precisely what must be computed/proved.",
            "List constraints and hidden assumptions.",
        ],
    ),
    StrategyCard(
        name="Represent the problem",
        instructions=[
            "Choose a representation: equations, diagram, graph, table, or coordinate model.",
            "Introduce variables for unknowns; name key quantities.",
            "Rewrite conditions as explicit algebra/logic statements.",
        ],
    ),
    StrategyCard(
        name="Simplify & solve a toy version",
        instructions=[
            "Solve a smaller/simpler case first (small n, special angles, small primes).",
            "Look for patterns and formulate a conjecture.",
            "Then generalize carefully.",
        ],
    ),
    StrategyCard(
        name="Work backward from the goal",
        instructions=[
            "Assume a candidate structure for the answer and derive necessary conditions.",
            "Try to reduce the target statement to known lemmas/identities.",
            "If proving, identify the final step and what would imply it.",
        ],
    ),
    StrategyCard(
        name="Look for invariants / monotonicity",
        instructions=[
            "Identify quantities that remain unchanged under allowed moves.",
            "If a process is involved, look for monotone measures or potentials.",
            "Use invariants/monotonicity to bound possibilities and force structure.",
        ],
    ),
    StrategyCard(
        name="Extremes & contradiction",
        instructions=[
            "Consider extreme or minimal counterexample arguments.",
            "Try bounding with max/min principles or choose an extremal element.",
            "Derive a contradiction or force a canonical configuration.",
        ],
    ),
    StrategyCard(
        name="Symmetry & normalization",
        instructions=[
            "Exploit symmetry: reorder, relabel, assume WLOG.",
            "Normalize by scaling/translation/rotation if allowed.",
            "Seek symmetric polynomials, cyclic sums, or invariant transformations.",
        ],
    ),
    StrategyCard(
        name="Compute, then prove",
        instructions=[
            "Use the Python tool to compute small cases or search candidates.",
            "Extract a clean conjecture (closed form, invariant, pattern).",
            "Then prove it rigorously.",
        ],
    ),
    StrategyCard(
        name="Check & verify",
        instructions=[
            "Verify edge cases and constraints (domains, integrality, positivity).",
            "Cross-check by an independent method or numeric sampling.",
            "Only output a boxed integer when you are confident.",
        ],
    ),
]


def select_strategy(attempt_index: int) -> StrategyCard:
    if not STRATEGY_CARDS:
        raise RuntimeError("No strategy cards configured")
    return STRATEGY_CARDS[int(attempt_index) % len(STRATEGY_CARDS)]


def render_strategy_card(card: StrategyCard) -> str:
    lines = [
        "Wickelgren-style strategy card (paraphrased):",
        f"- Focus: {card.name}",
    ]
    for item in card.instructions:
        lines.append(f"- {item}")
    return "\n".join(lines)


def augment_system_prompt(base_prompt: str, *, attempt_index: int) -> str:
    """Append a strategy card to the base system prompt."""
    card = select_strategy(attempt_index)
    strategy_text = render_strategy_card(card)
    if not base_prompt:
        return strategy_text
    return base_prompt.rstrip() + "\n\n" + strategy_text
