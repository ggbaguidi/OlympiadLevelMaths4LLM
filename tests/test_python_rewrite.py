from olympiad_llm.aimo3.python_rewrite import rewrite_python_tool_code


def test_rewrite_sp_valuation_inserts_import_and_rewrites_call():
    code = """# timeout: 5
x = 10
v = sp.valuation(x, 2)
"""
    out = rewrite_python_tool_code(code)
    assert "from sympy.polys.numberfields import prime_valuation" in out
    assert "sp.valuation" not in out
    assert "_aimo3_prime_valuation(x, 2)" in out


def test_rewrite_does_not_touch_strings_or_comments():
    code = """# sp.valuation(x,2) in a comment
s = "sp.valuation(x,2) in a string"
"""
    out = rewrite_python_tool_code(code)
    # No alias inserted because no real token match.
    assert "_aimo3_prime_valuation" not in out
    assert out == code


def test_rewrite_is_idempotent():
    code = """from sympy.polys.numberfields import prime_valuation as _aimo3_prime_valuation
x = 10
v = _aimo3_prime_valuation(x, 2)
"""
    out = rewrite_python_tool_code(code)
    assert out == code
