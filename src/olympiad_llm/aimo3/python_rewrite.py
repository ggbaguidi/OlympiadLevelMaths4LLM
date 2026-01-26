from __future__ import annotations

"""Best-effort rewrites for python-tool snippets.

Goal: make the python tool more robust to version-dependent APIs by applying
safe, minimal, *stdlib-only* rewrites before executing code in the sandbox.

This module is deliberately conservative:
- prefers token-based rewriting over naive string replacement
- avoids touching strings/comments

Current rewrites:
 Replace `sp.valuation` with a small compatibility wrapper `_aimo3_valuation`
  that tries the integer p-adic valuation first, then falls back to
  `prime_valuation` for numberfield contexts.
"""

import io
import tokenize
from typing import Iterable


_AIMO3_VALUATION_ALIAS = "_aimo3_valuation"
_AIMO3_INT_VALUATION_ALIAS = "_aimo3_int_valuation"
_AIMO3_PRIME_VALUATION_ALIAS = "_aimo3_prime_valuation"

_AIMO3_VALUATION_BLOCK = (
    f"from sympy.ntheory.factor_ import valuation as {_AIMO3_INT_VALUATION_ALIAS}\n"
    f"from sympy.polys.numberfields import prime_valuation as {_AIMO3_PRIME_VALUATION_ALIAS}\n"
    f"def {_AIMO3_VALUATION_ALIAS}(a, p):\n"
    "    \"\"\"Compatibility valuation helper.\n\n"
    "    - If p is an integer-like (int or sympy.Integer), use integer p-adic valuation.\n"
    "    - Otherwise, fall back to SymPy's numberfield prime_valuation.\n"
    "    \"\"\"\n"
    "    try:\n"
    "        import sympy as sp\n"
    "        if isinstance(p, (int, sp.Integer)):\n"
    f"            return {_AIMO3_INT_VALUATION_ALIAS}(a, int(p))\n"
    "    except Exception:\n"
    "        pass\n"
    f"    return {_AIMO3_PRIME_VALUATION_ALIAS}(a, p)\n"
)


def _is_name(tok: tokenize.TokenInfo, s: str) -> bool:
    return tok.type == tokenize.NAME and tok.string == s


def _is_op(tok: tokenize.TokenInfo, s: str) -> bool:
    return tok.type == tokenize.OP and tok.string == s


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
    """Replace token sequence `sp . valuation` with `_aimo3_valuation`."""

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
                    string=_AIMO3_VALUATION_ALIAS,
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


def _rewrite_prime_valuation_alias_to_wrapper(src: str) -> str:
    """Rewrite `_aimo3_prime_valuation` calls to `_aimo3_valuation`.

    Backward compatibility: older rewrites emitted `_aimo3_prime_valuation(...)`.
    That alias crashes on integer valuation use-cases; the wrapper dispatches.
    """

    if _AIMO3_PRIME_VALUATION_ALIAS not in src:
        return src

    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out: list[tokenize.TokenInfo] = []
    for t in toks:
        if t.type == tokenize.NAME and t.string == _AIMO3_PRIME_VALUATION_ALIAS:
            out.append(
                tokenize.TokenInfo(
                    type=tokenize.NAME,
                    string=_AIMO3_VALUATION_ALIAS,
                    start=t.start,
                    end=t.end,
                    line=t.line,
                )
            )
        else:
            out.append(t)
    return tokenize.untokenize(out)


def _count_top_level_commas(tokens: list[tokenize.TokenInfo], start_i: int, end_i: int) -> int:
    """Count commas at nesting depth 0 within (start_i, end_i) token indices."""

    depth = 0
    commas = 0
    for t in tokens[start_i:end_i]:
        if _is_op(t, "(") or _is_op(t, "[") or _is_op(t, "{"):
            depth += 1
        elif _is_op(t, ")") or _is_op(t, "]") or _is_op(t, "}"):
            depth = max(0, depth - 1)
        elif depth == 0 and _is_op(t, ","):
            commas += 1
    return commas


def _rewrite_sp_circle_three_points(src: str) -> str:
    """Rewrite `sp.Circle(A,B,C)` into `sp.Circle.from_three_points(A,B,C)`.

    Motivation: in some degenerate cases SymPy may return a non-Circle object (e.g. Segment2D)
    from `sp.Circle(p1,p2,p3)`. Using `from_three_points` gives a clearer failure mode.

    This rewrite is conservative:
    - only applies to the exact `sp.Circle(...)` call form
    - only when there are exactly 3 top-level args (2 commas)
    - does not touch strings/comments (token-based)
    - does not touch already-rewritten `sp.Circle.from_three_points(...)`
    """

    if "sp.Circle" not in src:
        return src

    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    # Use (tok_type, tok_string) pairs for untokenize to avoid relying on positional
    # metadata when injecting new tokens.
    out: list[tuple[int, str]] = []
    changed = False
    i = 0
    while i < len(toks):
        t = toks[i]

        # Match: sp . Circle
        if (
            _is_name(t, "sp")
            and i + 2 < len(toks)
            and _is_op(toks[i + 1], ".")
            and _is_name(toks[i + 2], "Circle")
        ):
            # If this is already `sp.Circle.from_three_points`, don't touch.
            if i + 4 < len(toks) and _is_op(toks[i + 3], ".") and _is_name(toks[i + 4], "from_three_points"):
                out.append((t.type, t.string))
                i += 1
                continue

            # Only rewrite when followed by a call: (...)
            j = i + 3
            if j < len(toks) and _is_op(toks[j], "("):
                # Find matching ')'
                depth = 0
                k = j
                while k < len(toks):
                    if _is_op(toks[k], "("):
                        depth += 1
                    elif _is_op(toks[k], ")"):
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1

                if k < len(toks) and _is_op(toks[k], ")"):
                    # tokens between j+1 and k are the arg list (with nesting)
                    commas = _count_top_level_commas(toks, j + 1, k)
                    if commas == 2:
                        # Emit: sp . Circle . from_three_points
                        out.extend(
                            [
                                (toks[i].type, toks[i].string),
                                (toks[i + 1].type, toks[i + 1].string),
                                (toks[i + 2].type, toks[i + 2].string),
                                (tokenize.OP, "."),
                                (tokenize.NAME, "from_three_points"),
                            ]
                        )
                        changed = True
                        i += 3
                        continue

        out.append((t.type, t.string))
        i += 1

    if not changed:
        return src
    return tokenize.untokenize(out)


def rewrite_python_tool_code(code: str) -> str:
    """Apply all configured rewrites to python-tool code."""

    src = str(code or "")
    rewritten = src

    if "sp.Circle" in rewritten:
        rewritten = _rewrite_sp_circle_three_points(rewritten)

    if "sp.valuation" in rewritten:
        rewritten = _rewrite_sp_valuation_tokens(rewritten)

    # Back-compat: older rewrite emitted `_aimo3_prime_valuation(...)`.
    # Only apply if the wrapper isn't already defined in the snippet; otherwise we risk
    # rewriting user-provided helper code and breaking idempotence.
    if _AIMO3_PRIME_VALUATION_ALIAS in rewritten and f"def {_AIMO3_VALUATION_ALIAS}" not in rewritten:
        rewritten = _rewrite_prime_valuation_alias_to_wrapper(rewritten)

    # If wrapper is referenced, ensure helper block is present.
    # IMPORTANT: avoid modifying user-provided wrapper definitions; if the function is
    # already defined (even with a different body), leave it untouched for idempotence.
    if _AIMO3_VALUATION_ALIAS in rewritten and f"def {_AIMO3_VALUATION_ALIAS}" not in rewritten:
        rewritten = _insert_import_after_header_lines(rewritten, _AIMO3_VALUATION_BLOCK)

    return rewritten
