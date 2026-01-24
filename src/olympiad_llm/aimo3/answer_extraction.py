from __future__ import annotations

import re
from dataclasses import dataclass


_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^}]*)\s*\}")
_BOXED_INT_RE = re.compile(r"\\boxed\s*\{\s*([0-9][0-9,]*)\s*\}")


@dataclass(frozen=True)
class AnswerExtractor:
    """Extract answers from model text.

    The AIMO scoring format expects an integer in a box: \boxed{n}.
    """

    aimo_lo: int = 0
    aimo_hi: int = 99999

    def extract_boxed_int(self, text: str) -> int | None:
        """Return the last valid \boxed{int} in [aimo_lo, aimo_hi], else None."""

        matches = list(_BOXED_INT_RE.findall(text or ""))
        if not matches:
            return None
        raw = matches[-1].replace(",", "")
        try:
            val = int(raw)
        except ValueError:
            return None
        if self.aimo_lo <= val <= self.aimo_hi:
            return val
        return None

    def extract_boxed_content(self, text: str) -> str | None:
        """Return the *content* of the last \boxed{...} (not necessarily int)."""

        matches = list(_BOXED_RE.findall(text or ""))
        if not matches:
            return None
        return matches[-1].strip()

    def normalize_final_answer_text(self, text: str) -> str:
        """If the answer contains \boxed{...}, return the boxed content; else return stripped text."""

        boxed = self.extract_boxed_content(text)
        return boxed if boxed is not None else (text or "").strip()
