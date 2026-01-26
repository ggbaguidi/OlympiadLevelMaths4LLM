import os


def test_profile_lean_overrides_defaults(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.setenv("AIMO3_PROFILE", "lean")
    # Ensure not explicitly set
    monkeypatch.delenv("AIMO3_ATTEMPTS", raising=False)
    monkeypatch.delenv("AIMO3_WORKERS", raising=False)
    monkeypatch.delenv("AIMO3_TURNS", raising=False)

    cfg = AIMO3Config.from_env()
    assert cfg.attempts == 4
    assert cfg.turns == 64
    assert cfg.workers >= 4


def test_profile_lean_does_not_override_explicit_env(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.setenv("AIMO3_PROFILE", "lean")
    monkeypatch.setenv("AIMO3_ATTEMPTS", "7")
    monkeypatch.setenv("AIMO3_TURNS", "10")
    monkeypatch.setenv("AIMO3_WORKERS", "9")

    cfg = AIMO3Config.from_env()
    assert cfg.attempts == 7
    assert cfg.turns == 10
    assert cfg.workers == 9


def test_turns_env_var(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.delenv("AIMO3_PROFILE", raising=False)
    monkeypatch.setenv("AIMO3_TURNS", "33")
    cfg = AIMO3Config.from_env()
    assert cfg.turns == 33


def test_code_first_phase_env_var(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.delenv("AIMO3_PROFILE", raising=False)
    monkeypatch.setenv("AIMO3_CODE_FIRST_PHASE_S", "300")

    cfg = AIMO3Config.from_env()
    assert cfg.code_first_phase_s == 300.0


def test_sandbox_reset_between_attempts_env_var(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.delenv("AIMO3_PROFILE", raising=False)
    monkeypatch.setenv("AIMO3_SANDBOX_RESET_BETWEEN_ATTEMPTS", "0")

    cfg = AIMO3Config.from_env()
    assert cfg.sandbox_reset_between_attempts is False


def test_disable_prompts_env_var(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.delenv("AIMO3_PROFILE", raising=False)
    monkeypatch.setenv("AIMO3_DISABLE_PROMPTS", "verification,analytic")

    cfg = AIMO3Config.from_env()
    assert cfg.disabled_prompts == "verification,analytic"


def test_second_stage_verify_enabled_env_var(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.delenv("AIMO3_PROFILE", raising=False)
    monkeypatch.setenv("AIMO3_SECOND_STAGE_VERIFY_ENABLED", "0")

    cfg = AIMO3Config.from_env()
    assert cfg.second_stage_verify_enabled is False


def test_trace_env_config_env_vars(monkeypatch):
    from olympiad_llm.aimo3.config import AIMO3Config

    monkeypatch.delenv("AIMO3_PROFILE", raising=False)
    monkeypatch.setenv("AIMO3_TRACE_ENV", "1")
    monkeypatch.setenv("AIMO3_TRACE_ENV_PACKAGES", "sympy,numpy,mpmath,jupyter_client")

    cfg = AIMO3Config.from_env()
    assert cfg.trace_env_enabled is True
    assert cfg.trace_env_packages == "sympy,numpy,mpmath,jupyter_client"

