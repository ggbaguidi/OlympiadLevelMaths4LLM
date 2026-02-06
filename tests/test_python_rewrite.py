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


def test_rewrite_injects_missing_combinations_import():
    """Auto-inject 'from itertools import combinations' when combinations is used."""
    code = """for a, b in combinations(range(10), 2):
    print(a, b)
"""
    out = rewrite_python_tool_code(code)
    assert "from itertools import combinations" in out


def test_rewrite_injects_missing_counter_import():
    """Auto-inject 'from collections import Counter' when Counter is used."""
    code = """c = Counter([1, 2, 2, 3])
print(c.most_common())
"""
    out = rewrite_python_tool_code(code)
    assert "from collections import Counter" in out


def test_rewrite_injects_multiple_missing_imports():
    """Auto-inject multiple imports when multiple bare names are used."""
    code = """for a, b in combinations(range(10), 2):
    if gcd(a, b) == 1:
        print(a, b)
"""
    out = rewrite_python_tool_code(code)
    assert "from itertools import combinations" in out
    assert "from math import gcd" in out


def test_rewrite_does_not_inject_for_attribute_access():
    """Don't inject import for attribute access like itertools.combinations."""
    code = """import itertools
for a, b in itertools.combinations(range(10), 2):
    print(a, b)
"""
    out = rewrite_python_tool_code(code)
    # Should not inject since it's used as attribute, not bare name.
    assert out == code


def test_rewrite_does_not_inject_for_defined_names():
    """Don't inject import for names that are defined in the code."""
    code = """def combinations(n, k):
    return n * k
print(combinations(5, 3))
"""
    out = rewrite_python_tool_code(code)
    # Should not inject since 'combinations' is defined in the code.
    assert "from itertools import combinations" not in out


def test_rewrite_does_not_inject_if_already_imported():
    """Don't inject if import already exists."""
    code = """from itertools import combinations
for a, b in combinations(range(10), 2):
    print(a, b)
"""
    out = rewrite_python_tool_code(code)
    # Should not duplicate import.
    assert out.count("from itertools import combinations") == 1
