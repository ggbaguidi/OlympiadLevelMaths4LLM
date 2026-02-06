from olympiad_llm.aimo3.config import AIMO3Config


def test_recent_python_outputs_in_conclusion_defaults():
    cfg = AIMO3Config.from_env()
    assert cfg.recent_python_outputs_in_conclusion_enabled is True
    assert cfg.recent_python_outputs_in_conclusion_n == 5
    assert int(cfg.recent_python_outputs_in_conclusion_max_chars) > 0


def test_recent_python_outputs_in_conclusion_env_parsing(monkeypatch):
    monkeypatch.setenv("AIMO3_RECENT_PYTHON_OUTPUTS_IN_CONCLUSION_ENABLED", "0")
    monkeypatch.setenv("AIMO3_RECENT_PYTHON_OUTPUTS_IN_CONCLUSION_N", "9")
    monkeypatch.setenv("AIMO3_RECENT_PYTHON_OUTPUTS_IN_CONCLUSION_MAX_CHARS", "123")
    cfg = AIMO3Config.from_env()
    assert cfg.recent_python_outputs_in_conclusion_enabled is False
    assert cfg.recent_python_outputs_in_conclusion_n == 9
    assert cfg.recent_python_outputs_in_conclusion_max_chars == 123
