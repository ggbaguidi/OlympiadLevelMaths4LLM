from olympiad_llm.aimo3.config import AIMO3Config
from olympiad_llm.aimo3.decoding import temperature_for_attempt


def test_temperature_for_attempt_priorities():
    cfg = AIMO3Config(
        temperature=0.9,
        temperature_exploration=0.95,
        temperature_main=0.7,
        temperature_code=0.6,
        temperature_verification=0.2,
        temperature_formatting=0.1,
        exploration_attempts=2,
    )

    # Formatting dominates
    assert temperature_for_attempt(cfg=cfg, attempt_index=0, attempt_tag="recovery|card=format_only") == 0.1

    # Verification dominates
    assert temperature_for_attempt(cfg=cfg, attempt_index=0, attempt_tag="verification|pack=x") == 0.2
    assert temperature_for_attempt(cfg=cfg, attempt_index=100, attempt_tag="second_stage_verify:cand=123") == 0.2
    assert temperature_for_attempt(cfg=cfg, attempt_index=100, attempt_tag="tiebreak|variant=micro_tool") == 0.2

    # Code-first
    assert temperature_for_attempt(cfg=cfg, attempt_index=5, attempt_tag="code_first|pack=generic") == 0.6

    # Exploration for early attempts
    assert temperature_for_attempt(cfg=cfg, attempt_index=0, attempt_tag="standard|pack=generic") == 0.95
    assert temperature_for_attempt(cfg=cfg, attempt_index=1, attempt_tag="analytic|pack=generic") == 0.95

    # Main after exploration
    assert temperature_for_attempt(cfg=cfg, attempt_index=2, attempt_tag="standard|pack=generic") == 0.7


def test_temperature_for_attempt_falls_back_to_default_when_none():
    cfg = AIMO3Config(
        temperature=0.8,
        temperature_exploration=None,
        temperature_main=None,
        temperature_code=None,
        temperature_verification=None,
        temperature_formatting=None,
        exploration_attempts=1,
    )

    assert temperature_for_attempt(cfg=cfg, attempt_index=0, attempt_tag="standard") == 0.8
    assert temperature_for_attempt(cfg=cfg, attempt_index=10, attempt_tag="verification") == 0.8
