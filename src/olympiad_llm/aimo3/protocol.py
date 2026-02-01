"""Prompt protocol helpers for AIMO-3.

This enforces a stable solve loop inside a single attempt:
- push toward code-verified solutions
- require verification before boxing
- produce \boxed{n} with verified answer

This is *in addition* to the outer controller (multi-attempt + voting).
"""

from __future__ import annotations


def protocol_suffix() -> str:
    # Ultra-short protocol: maximize time spent on actual computation
    return (
        "\n\n"
        "RULES:\n"
        "• Use Python to compute/verify. Don't do arithmetic by hand.\n"
        "• Test your answer on small cases BEFORE boxing it.\n"
        "• Final answer: \\boxed{n} where n is an integer in [0,99999].\n"
    )


def with_protocol(system_prompt: str) -> str:
    return (system_prompt or "").rstrip() + protocol_suffix()
