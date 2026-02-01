from olympiad_llm.aimo3.config import AIMO3Config
from olympiad_llm.aimo3.solver import AIMO3Solver


def test_enabled_prompt_specs_filters_disabled_names():
    cfg = AIMO3Config(disabled_prompts="verification,analytic")
    specs = AIMO3Solver._enabled_prompt_specs(cfg)  # noqa: SLF001
    names = [n for (n, _p) in specs]
    assert "verification" not in names
    assert "analytic" not in names
    assert "standard" in names


def test_enabled_prompt_specs_never_empty_even_if_all_disabled():
    cfg = AIMO3Config(disabled_prompts="standard,code_first,analytic,verification,small_cases,sanity")
    specs = AIMO3Solver._enabled_prompt_specs(cfg)  # noqa: SLF001
    assert specs
    assert specs[0][0] == "standard"


def test_enabled_prompt_specs_ignores_unknown_names():
    cfg = AIMO3Config(disabled_prompts="not_a_prompt")
    specs = AIMO3Solver._enabled_prompt_specs(cfg)  # noqa: SLF001
    names = [n for (n, _p) in specs]
    assert set(names) == {"standard", "code_first", "analytic", "verification", "small_cases", "sanity"}
