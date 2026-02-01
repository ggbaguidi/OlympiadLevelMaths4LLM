"""Prompt protocol helpers for AIMO-3.

This enforces a stable solve loop inside a single attempt:
- push toward code-verified solutions
- require verification before boxing
- produce \boxed{n} with verified answer

This is *in addition* to the outer controller (multi-attempt + voting).
"""

from __future__ import annotations


def protocol_suffix() -> str:
    # Concise protocol with problem-parsing guidance to avoid misinterpretation
    return (
        "\n\n"
        "RULES:\n"
        "1. PARSE FIRST: Restate the problem's key definitions, constraints, and what's being counted/computed. Check for edge cases in the wording.\n"
        "2. COMPUTE: Use Python—don't do arithmetic by hand. Test on small cases first.\n"
        "3. VERIFY: Confirm your interpretation matches the problem before boxing.\n"
        "4. Final answer: \\boxed{n} where n ∈ [0,99999].\n"
    )


def with_protocol(system_prompt: str) -> str:
    return (system_prompt or "").rstrip() + protocol_suffix()
