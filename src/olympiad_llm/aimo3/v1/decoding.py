"""Decoding policy helpers.

These functions are intentionally problem-agnostic: they select decoding
parameters based on the *role* of an attempt (exploration vs verification, etc).
"""

from __future__ import annotations

from .config import AIMO3Config


def temperature_for_attempt(
    *, cfg: AIMO3Config, attempt_index: int, attempt_tag: str | None
) -> float:
    """Choose a temperature based on attempt role.

    Priority (most specific first):
    - formatting recovery
    - verification / tie-break / second-stage verification
    - code-first
    - exploration (first N attempts)
    - main

    Falls back to cfg.temperature.
    """

    tag = str(attempt_tag or "")

    def _fallback(x: float | None) -> float:
        return float(cfg.temperature if x is None else x)

    # Formatting-only attempts should be very low variance.
    if "format_only" in tag or "formatting" in tag:
        return _fallback(cfg.temperature_formatting)

    # Verification/tie-break attempts should be low variance.
    if (
        "verification" in tag
        or tag.startswith("second_stage_verify")
        or tag.startswith("tiebreak")
        or "cand=" in tag
        and "second_stage_verify" in tag
    ):
        return _fallback(cfg.temperature_verification)

    # Recovery attempts without tool are also closer to verification style.
    if "recovery" in tag and "variant=no_tool" in tag:
        return _fallback(cfg.temperature_verification)

    # Code-first attempts generally benefit from slightly lower temperature.
    if "code_first" in tag:
        return _fallback(cfg.temperature_code)

    # Exploration early attempts.
    if int(attempt_index) < int(getattr(cfg, "exploration_attempts", 0) or 0):
        return _fallback(cfg.temperature_exploration)

    return _fallback(cfg.temperature_main)
