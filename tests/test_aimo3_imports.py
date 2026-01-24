import pytest


def test_aimo3_lightweight_imports():
    # These must import without heavy optional dependencies.
    import olympiad_llm.aimo3 as aimo3  # noqa: F401
    from olympiad_llm.aimo3.config import AIMO3Config
    from olympiad_llm.aimo3.prompts import TIR_PROMPTS

    cfg = AIMO3Config.from_env()
    assert isinstance(cfg.model_path, str)
    assert len(TIR_PROMPTS) >= 4


def test_aimo3_solver_requires_optional_deps():
    # Importing the solver module is OK, but constructing it should error
    # if optional deps are not installed in this environment.
    from olympiad_llm.aimo3.config import AIMO3Config
    from olympiad_llm.aimo3.errors import OptionalDependencyError

    try:
        from olympiad_llm.aimo3.solver import AIMO3Solver
    except OptionalDependencyError:
        pytest.skip("Optional deps missing at import time")

    with pytest.raises(Exception):
        _ = AIMO3Solver(AIMO3Config(model_path="/nonexistent"))
