# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""Compact reasoning framework for olympiad-style math problems.

Inspired by David Kelley's emphasis on clarity, structure, evidence, and
intellectual honesty. This module adapts those ideas to exact mathematical
problem solving for v2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningStep:
    name: str
    instruction: str


MATH_REASONING_STEPS: tuple[ReasoningStep, ...] = (
    ReasoningStep(
        name="clarify_target",
        instruction=(
            "State exactly what must be computed, proved, or optimized; define the "
            "output and all variable/domain restrictions."
        ),
    ),
    ReasoningStep(
        name="extract_premises",
        instruction=(
            "Translate the givens into exact constraints, equations, recurrences, "
            "invariants, symmetries, or counting rules."
        ),
    ),
    ReasoningStep(
        name="choose_reasoning_mode",
        instruction=(
            "Classify the problem type first (counting, divisibility, algebra, "
            "geometry, recurrence, probability, optimization) and choose the strongest exact tools."
        ),
    ),
    ReasoningStep(
        name="plan_exact_method",
        instruction=(
            "Pick one primary exact route and one cheap cross-check. Prefer formulas, "
            "bijections, recurrences, modular arguments, valuations, or exact algebra over guesses."
        ),
    ),
    ReasoningStep(
        name="test_small_cases",
        instruction=(
            "Use tiny cases only to falsify or refine a conjecture, never as final proof. "
            "Check edge cases and conserved quantities."
        ),
    ),
    ReasoningStep(
        name="execute_exactly",
        instruction=(
            "Carry out the chosen method with exact arithmetic. For counting or divisibility, "
            "compute exact products, recurrences, factorizations, and p-adic valuations."
        ),
    ),
    ReasoningStep(
        name="challenge_result",
        instruction=(
            "Try to break the candidate answer by an alternative derivation, parity/mod checks, "
            "or tiny brute force. Reject any step that depends on an unsupported assumption."
        ),
    ),
    ReasoningStep(
        name="finalize",
        instruction=(
            "Only after the checks pass, return the final integer as \\boxed{n}."
        ),
    ),
)


def infer_reasoning_focus(problem_text: str | None) -> list[str]:
    text = (problem_text or "").lower()
    hints: list[str] = []

    if any(
        k in text
        for k in (
            "divides",
            "divisible",
            "remainder",
            "mod",
            "factor of 10",
            "valuation",
            "trailing",
        )
    ):
        hints.append(
            "Divisibility focus: use modular arithmetic and exact p-adic valuations; avoid decimal heuristics."
        )

    if any(
        k in text
        for k in (
            "count",
            "number of",
            "ways",
            "ordering",
            "arrangement",
            "permutation",
            "tournament",
            "round",
            "race",
        )
    ):
        hints.append(
            "Counting focus: model the exact combinatorial structure first; prefer recurrences, bijections, symmetries, and exact counts."
        )

    if any(k in text for k in ("probability", "expected", "random", "chance")):
        hints.append(
            "Probability focus: count exact favorable and total cases; keep expressions rational until the end."
        )

    if any(
        k in text for k in ("sequence", "recurrence", "recursive", "term", "f_n", "a_n")
    ):
        hints.append(
            "Recurrence focus: derive the exact recurrence or invariant before computing many terms."
        )

    if any(
        k in text
        for k in (
            "triangle",
            "circle",
            "angle",
            "tangent",
            "chord",
            "geometry",
            "perpendicular",
        )
    ):
        hints.append(
            "Geometry focus: use exact geometric relations first; only introduce coordinates if they simplify the structure."
        )

    return hints[:2]


def render_reasoning_framework(problem_text: str | None = None) -> str:
    lines = [
        "[META_REASONING_FRAMEWORK]",
        "Work through these steps internally. Use them as a checklist, not as extra prose.",
        "Good reasoning here means clear premises, exact methods, and active error-checking.",
    ]

    focus = infer_reasoning_focus(problem_text)
    if focus:
        lines.append("Focus:")
        for item in focus:
            lines.append(f"- {item}")

    for idx, step in enumerate(MATH_REASONING_STEPS, start=1):
        lines.append(f"{idx}. {step.instruction}")

    lines.append("[/META_REASONING_FRAMEWORK]")
    return "\n".join(lines)


def augment_prompt_with_reasoning_framework(
    base_prompt: str, problem_text: str | None = None
) -> str:
    framework = render_reasoning_framework(problem_text)
    return framework if not base_prompt else base_prompt.rstrip() + "\n\n" + framework
