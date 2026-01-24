from olympiad_llm.aimo3.prompts import ENHANCED_TOOL_INSTRUCTION


def test_tool_instruction_mentions_mpmath_alias():
    # The model should be nudged toward using a stable alias.
    assert "mpmath" in ENHANCED_TOOL_INSTRUCTION
    assert "mp" in ENHANCED_TOOL_INSTRUCTION
