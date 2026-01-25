from olympiad_llm.aimo3.config import AIMO3Config
from olympiad_llm.aimo3.prompts import ENHANCED_TOOL_INSTRUCTION


def test_preference_prompt_mentions_ortools():
    assert "ortools" in AIMO3Config.preference_prompt


def test_tool_instruction_mentions_ortools():
    assert "ortools" in ENHANCED_TOOL_INSTRUCTION
    assert "cp_model" in ENHANCED_TOOL_INSTRUCTION
