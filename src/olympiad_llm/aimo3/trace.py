from __future__ import annotations

"""Lightweight JSONL tracing for AIMO-3 runs.

This is intentionally dependency-free (stdlib only) and safe by default:
- does not record full problem text unless explicitly enabled.

The goal is to make tuning measurable: why did we pick an answer, what did the
ranking look like, how much time did we spend, etc.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any


def stable_problem_id(problem_text: str) -> str:
    """Return a stable short id for a problem prompt."""

    h = hashlib.sha1((problem_text or "").encode("utf-8"), usedforsecurity=False)
    return h.hexdigest()[:12]


@dataclass
class TraceRecorder:
    enabled: bool
    path: str
    include_problem_text: bool = False

    def record(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return

        # Ensure output directory exists.
        p = str(self.path or "aimo3_trace.jsonl")
        d = os.path.dirname(p)
        if d:
            os.makedirs(d, exist_ok=True)

        # Normalize and add minimal metadata.
        payload = dict(event)
        payload.setdefault("ts", time.time())

        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
