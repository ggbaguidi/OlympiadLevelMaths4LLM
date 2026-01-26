from __future__ import annotations

"""Best-effort rewrites for python-tool snippets.

Goal: make the python tool more robust to version-dependent APIs by applying
safe, minimal, *stdlib-only* rewrites before executing code in the sandbox.

This module is deliberately conservative:
- prefers token-based rewriting over naive string replacement
- avoids touching strings/comments

Current rewrites:
- Replace `sp.valuation` with `prime_valuation` (SymPy numberfields helper) and
  inject a stable alias import.

Notes:
- `prime_valuation` has different semantics than integer p-adic valuation.
  This rewrite is meant to prevent immediate AttributeError crashes when models
  use `sp.valuation(...)` in numberfield contexts.
"""

import io
import tokenize
from typing import Iterable


_PRIME_VAL_ALIAS = "_aimo3_prime_valuation"
_PRIME_VAL_IMPORT = f"from sympy.polys.numberfields import prime_valuation as {_PRIME_VAL_ALIAS}"


def _insert_import_after_header_lines(code: str, import_line: str) -> str:
    """Insert `import_line` after leading blank/comment lines.

    This keeps common headers like:
    - encoding comment
    - `# timeout: N`
    - other comments

    If the import already exists (exact line match), return unchanged.
    """

    if import_line in code:
        return code

    lines = code.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == "" or s.startswith("#"):
            i += 1
            continue
        break

    insert = import_line + ("\n" if (not import_line.endswith("\n")) else "")
    return "".join(lines[:i] + [insert] + lines[i:])


def _rewrite_sp_valuation_tokens(src: str) -> str:
    """Replace token sequence `sp . valuation` with `_aimo3_prime_valuation`."""

    # Tokenize / untokenize preserves formatting reasonably well.
    out_tokens: list[tokenize.TokenInfo] = []
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))

    i = 0
    while i < len(toks):
        t = toks[i]

        # Look for NAME 'sp' OP '.' NAME 'valuation'
        if (
            t.type == tokenize.NAME
            and t.string == "sp"
            and i + 2 < len(toks)
            and toks[i + 1].type == tokenize.OP
            and toks[i + 1].string == "."
            and toks[i + 2].type == tokenize.NAME
            and toks[i + 2].string == "valuation"
        ):
            # Replace the 3-token sequence with a single NAME token.
            out_tokens.append(
                tokenize.TokenInfo(
                    type=tokenize.NAME,
                    string=_PRIME_VAL_ALIAS,
                    start=t.start,
                    end=toks[i + 2].end,
                    line=t.line,
                )
            )
            i += 3
            continue

        out_tokens.append(t)
        i += 1

    return tokenize.untokenize(out_tokens)


def rewrite_python_tool_code(code: str) -> str:
    """Apply all configured rewrites to python-tool code."""

    src = str(code or "")
    if "sp.valuation" not in src:
        return src

    rewritten = _rewrite_sp_valuation_tokens(src)

    # If we actually introduced the alias usage, inject the import.
    if _PRIME_VAL_ALIAS in rewritten and _PRIME_VAL_IMPORT not in rewritten:
        rewritten = _insert_import_after_header_lines(rewritten, _PRIME_VAL_IMPORT)

    return rewritten
