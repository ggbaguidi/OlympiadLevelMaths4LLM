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
