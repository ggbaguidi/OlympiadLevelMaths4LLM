"""Prompt protocol helpers for AIMO-3.

This enforces a stable solve loop inside a single attempt:
- make subgoals explicit (lemmas)
- run checks
- only produce \boxed{n} when confident

This is *in addition* to the outer controller (multi-attempt + voting).
"""

from __future__ import annotations


def protocol_suffix() -> str:
    # Keep it short: token-efficient and hard to ignore.
    return (
        "\n\n"
        "Protocol (follow strictly):\n"
        "1) First, restate the goal and list <=3 subgoals/lemmas.\n"
        "2) If computation/search helps, use the Python tool (show the check).\n"
        "3) Before finalizing, run a final Python check or sanity test.\n"
        "4) If you are not fully confident, output NOBOX (do not output any \\boxed{...}).\n"
        "5) If confident, output exactly one final line: \\boxed{n} with integer n in [0,99999].\n"
    )


def with_protocol(system_prompt: str) -> str:
    return (system_prompt or "").rstrip() + protocol_suffix()
