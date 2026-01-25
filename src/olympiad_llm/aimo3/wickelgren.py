from __future__ import annotations

"""Wickelgren-inspired problem-solving strategies.

This module provides *paraphrased* strategy checklists inspired by classical
math problem-solving heuristics (including Wickelgren-style guidance), without
reproducing any book text.

Goal: reduce prompt brittleness by giving the model a concrete, varied
"strategy card" each attempt (understand → explore → plan → execute → check).
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyCard:
    name: str
    instructions: list[str]


GENERIC_STRATEGY_CARDS: list[StrategyCard] = [
    StrategyCard(
        name="Understand & restate",
        instructions=[
            "Restate the problem in your own words and define every symbol.",
            "State precisely what must be computed/proved.",
            "List constraints and hidden assumptions (domain, integrality, bounds).",
        ],
    ),
    StrategyCard(
        name="Represent the problem",
        instructions=[
            "Propose 2 representations (equations/diagram/graph/table/coordinates) and pick one.",
            "Introduce variables for unknowns; name key quantities.",
            "Rewrite conditions as explicit algebra/logic statements.",
        ],
    ),
    StrategyCard(
        name="Simplify & solve a toy version",
        instructions=[
            "Solve a smaller/simpler case first (small n, special angles, small primes).",
            "Look for patterns and formulate a conjecture.",
            "Then generalize carefully: state what changes and what stays invariant.",
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
            "Extract a clean conjecture (closed form, invariant, pattern) and state it explicitly.",
            "Then prove it rigorously (explain why the computation is evidence, not proof).",
        ],
    ),
    StrategyCard(
        name="Check & verify",
        instructions=[
            "Verify edge cases and constraints (domains, integrality, positivity).",
            "Cross-check by an independent method or numeric sampling; try to refute your result.",
            "If checks are inconclusive, do not guess.",
        ],
    ),
]


FE_COMBI_STRATEGY_CARDS: list[StrategyCard] = [
    StrategyCard(
        name="Functional equation sanity checks",
        instructions=[
            "Extract immediate consequences: plug in 0, 1, -1, and symmetric inputs like (x,x), (x,0), (0,y).",
            "Check injectivity/surjectivity patterns and common forms (additive, multiplicative, linear, constant).",
            "Generate 2 candidate forms; quickly try to disprove them (counterexample search / constraints).",
            "Use the Python tool to test candidate forms on random small integers/rationals when appropriate.",
        ],
    ),
    StrategyCard(
        name="Combinatorial invariants & orbits",
        instructions=[
            "Look for an invariant or potential function under the described operation.",
            "If a group action/symmetry is present, consider orbit decomposition or normalization.",
            "Try extremal arguments: pick a minimal/maximal object and derive forced structure.",
        ],
    ),
    StrategyCard(
        name="Double counting & encoding",
        instructions=[
            "Try a double-counting viewpoint (count the same set in two ways).",
            "Encode objects as sequences/graphs and use injections/surjections to compare sizes.",
            "If the answer is an integer, try modular constraints or parity/valuation invariants.",
            "As a sanity check, compute small cases to confirm the combinatorial model matches the statement.",
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


def select_strategy(attempt_index: int, *, pack: str = "generic") -> StrategyCard:
    p = PACKS.get(pack, PACKS["generic"])
    if not p.cards:
        raise RuntimeError("No strategy cards configured")
    return p.cards[int(attempt_index) % len(p.cards)]


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
    lines = [
        "Wickelgren-style strategy card (paraphrased):",
        f"- Focus: {card.name}",
        "- Micro-rules: propose 2 angles → pick 1; include a toy check; don't claim verification without a check.",
    ]
    for item in card.instructions:
        lines.append(f"- {item}")
    return "\n".join(lines)


def augment_system_prompt_with_meta(
    base_prompt: str,
    *,
    attempt_index: int,
    problem_text: str | None = None,
    mode: str = "round_robin",
    enabled_packs: str | list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Append a strategy card and return (prompt, meta).

    Meta includes the selected pack and card name for attempt tagging.
    """

    pack = select_strategy_pack(
        attempt_index=int(attempt_index),
        problem_text=problem_text,
        mode=mode,
        enabled_packs=enabled_packs,
    )
    card = select_strategy(int(attempt_index), pack=pack)
    strategy_text = render_strategy_card(card)
    out = strategy_text if not base_prompt else base_prompt.rstrip() + "\n\n" + strategy_text
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
