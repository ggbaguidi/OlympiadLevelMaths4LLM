from __future__ import annotations

import re
from dataclasses import dataclass


_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^}]*)\s*\}")
_BOXED_INT_RE = re.compile(r"\\boxed\s*\{\s*([0-9][0-9,]*)\s*\}")

# Fallback patterns when the model forgets boxing.
_FINAL_INT_HINT_RE = re.compile(
    r"(?:final\s+answer|answer|ans)\s*(?:is|=|:)?\s*([0-9][0-9,]*)",
    flags=re.IGNORECASE,
)
_ANY_INT_RE = re.compile(r"\b([0-9][0-9,]*)\b")


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

    def extract_int_fallback(self, text: str) -> int | None:
        """Return a likely final integer answer even if unboxed.

        Strategy (general, conservative):
        1) Prefer integers that appear near an "answer" hint.
        2) Otherwise fall back to the last integer in-range in the text.

        This helps recover from formatting failures without changing the model prompt.
        """

        t = text or ""
        hinted = list(_FINAL_INT_HINT_RE.findall(t))
        candidates = hinted if hinted else list(_ANY_INT_RE.findall(t))
        if not candidates:
            return None

        # Choose the *last* in-range integer (closest to the end of the response).
        for raw in reversed(candidates):
            raw2 = raw.replace(",", "")
            try:
                val = int(raw2)
            except ValueError:
                continue
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
