from olympiad_llm.aimo3.python_rewrite import rewrite_python_tool_code


def test_rewrite_sp_valuation_inserts_import_and_rewrites_call():
    code = """# timeout: 5
x = 10
v = sp.valuation(x, 2)
"""
    out = rewrite_python_tool_code(code)
    assert "from sympy.polys.numberfields import prime_valuation" in out
    assert "def _aimo3_valuation" in out
    assert "sp.valuation" not in out
    assert "_aimo3_valuation(x, 2)" in out


def test_rewrite_does_not_touch_strings_or_comments():
    code = """# sp.valuation(x,2) in a comment
s = "sp.valuation(x,2) in a string"
"""
    out = rewrite_python_tool_code(code)
    # No alias inserted because no real token match.
    assert "_aimo3_valuation" not in out
    assert out == code


def test_rewrite_is_idempotent():
    code = """from sympy.ntheory.factor_ import valuation as _aimo3_int_valuation
from sympy.polys.numberfields import prime_valuation as _aimo3_prime_valuation
def _aimo3_valuation(a, p):
    try:
        import sympy as sp
        if isinstance(p, (int, sp.Integer)):
            return _aimo3_int_valuation(a, int(p))
    except Exception:
        pass
    return _aimo3_prime_valuation(a, p)
x = 10
v = _aimo3_valuation(x, 2)
"""
    out = rewrite_python_tool_code(code)
    assert out == code


def test_rewrite_sp_circle_three_points():
    code = """circ1 = sp.Circle(A, E, F)
O1 = circ1.center
"""
    out = rewrite_python_tool_code(code)
    assert "def _aimo3_circle3" in out
    norm = "".join(out.split())
    assert "_aimo3_circle3(A,E,F)" in norm
    assert "sp.Circle(A,E,F)" not in norm


def test_rewrite_sp_circle_does_not_touch_two_arg_circle():
    code = """# Circle by center+radius should not be rewritten
circ = sp.Circle(A, 3)
"""
    out = rewrite_python_tool_code(code)
    assert out == code
