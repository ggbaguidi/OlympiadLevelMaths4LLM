from olympiad_llm.aimo3.config import AIMO3Config
from olympiad_llm.aimo3.prompts import ENHANCED_TOOL_INSTRUCTION


def test_preference_prompt_mentions_lean_toolchain_optionally():
    # We only promise Lean4 when installed, but the prompt should mention the possibility.
    assert "lean" in AIMO3Config.preference_prompt.lower()
    assert "lake" in AIMO3Config.preference_prompt.lower()


def test_tool_instruction_mentions_lean_toolchain_optionally():
    assert "lean" in ENHANCED_TOOL_INSTRUCTION.lower()
    assert "lake" in ENHANCED_TOOL_INSTRUCTION.lower()
    assert "subprocess" in ENHANCED_TOOL_INSTRUCTION.lower()
