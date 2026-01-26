from olympiad_llm.aimo3.config import AIMO3Config


def test_finalize_answer_defaults():
    cfg = AIMO3Config.from_env()
    assert cfg.finalize_answer_enabled is True
    assert int(cfg.finalize_answer_max_tokens) > 0
    assert float(cfg.finalize_answer_min_remaining_s) >= 0.0


def test_finalize_answer_env_parsing(monkeypatch):
    monkeypatch.setenv("AIMO3_FINALIZE_ANSWER_ENABLED", "0")
    monkeypatch.setenv("AIMO3_FINALIZE_ANSWER_MAX_TOKENS", "77")
    monkeypatch.setenv("AIMO3_FINALIZE_ANSWER_MIN_REMAINING_S", "4.5")
    cfg = AIMO3Config.from_env()
    assert cfg.finalize_answer_enabled is False
    assert cfg.finalize_answer_max_tokens == 77
    assert cfg.finalize_answer_min_remaining_s == 4.5
