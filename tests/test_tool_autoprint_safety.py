from olympiad_llm.aimo3.solver import AIMO3Tool


def test_autoprint_does_not_break_function_definition():
    code = """def reflect_point_across_line(P, line):
    return P.reflect(line)
"""
    # Must not rewrite the last line into print(return ...)
    out = AIMO3Tool._ensure_last_print(code)  # noqa: SLF001
    assert "print(return" not in out
    assert out.strip().startswith("def reflect_point_across_line")
    assert "return P.reflect(line)" in out


def test_autoprint_wraps_single_expression():
    code = "2 + 3"
    out = AIMO3Tool._ensure_last_print(code)  # noqa: SLF001
    assert out.strip() == "print(2 + 3)"


def test_autoprint_wraps_last_top_level_expression_in_multiline():
    code = """x = 10
x + 1
"""
    out = AIMO3Tool._ensure_last_print(code)  # noqa: SLF001
    assert out.splitlines()[-1].strip() == "print(x + 1)"


def test_autoprint_does_not_touch_indented_last_line():
    code = """for i in range(2):
    i
"""
    out = AIMO3Tool._ensure_last_print(code)  # noqa: SLF001
    assert out == code


def test_autoprint_handles_trailing_comments():
    """Trailing comments must not end up inside print() which causes SyntaxError."""
    # Without this fix: print(table[:10] # just test) - broken because # hides closing paren
    code = "table[:10] # just test"
    out = AIMO3Tool._ensure_last_print(code)  # noqa: SLF001
    # Comment should be moved outside: print(table[:10])  # just test
    assert out == "print(table[:10])  # just test"


def test_autoprint_preserves_hash_in_strings():
    """Hash inside strings is not a comment, should not be extracted."""
    # Assignment, so should not be wrapped at all
    code = 's = "hello # world"'
    out = AIMO3Tool._ensure_last_print(code)  # noqa: SLF001
    assert out == code
