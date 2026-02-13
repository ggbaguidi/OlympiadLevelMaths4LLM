# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttemptStats:
    """Lightweight stats about a single attempt/run."""

    token_count: int = 0
    python_calls: int = 0
    python_errors: int = 0
    timeout_count: int = 0
    deadline_exceeded: bool = False

    mean_entropy: float = float("inf")

    verification_marker_found: bool | None = None

    last_error: str | None = None

    @property
    def tool_verified(self) -> bool:
        if self.verification_marker_found:
            return True

        if self.python_calls <= 0 or self.python_errors > 0:
            return False

        if self.verification_marker_found is None:
            return True
        return bool(self.verification_marker_found)

    @property
    def had_timeout(self) -> bool:
        return self.timeout_count > 0


@dataclass(frozen=True)
class AttemptResult:
    """Shared schema for an attempt."""

    attempt: int
    answer: int | str | None
    stats: AttemptStats = AttemptStats()
    output_text: str | None = None
    tag: str | None = None
