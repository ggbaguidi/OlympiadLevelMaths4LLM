# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name
from __future__ import annotations

import contextlib
from .require import _require_harmony


# ---------------------------------------------------------------------------
# Verification prompt strategies
# ---------------------------------------------------------------------------
# Each strategy is a short, focused prompt that tells the model to check
# a candidate answer via a specific method.  The model gets tool access
# (Python sandbox) so it can compute.

VERIFY_SUBSTITUTION = (
    "A proposed answer to the following problem is {answer}.\n\n"
    "Problem: {problem}\n\n"
    "Your task: VERIFY this answer by substituting it back into the problem conditions. "
    "Use Python to compute every step explicitly — do NOT skip computation. "
    "After your analysis, you MUST print your conclusion as a Python print statement:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer inside \\boxed{{}}.\n"
    "IMPORTANT: You MUST call print() with the VERDICT line — this is required."
)

VERIFY_SMALL_CASES = (
    "A proposed answer to the following problem is {answer}.\n\n"
    "Problem: {problem}\n\n"
    "Your task: CHECK this answer by computing small cases and boundary cases in Python. "
    "Build the answer from scratch using brute-force enumeration or exhaustive search on "
    "small instances, then compare with the proposed answer. "
    "After your analysis, you MUST print your conclusion as a Python print statement:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer inside \\boxed{{}}.\n"
    "IMPORTANT: You MUST call print() with the VERDICT line — this is required."
)

VERIFY_ALTERNATIVE = (
    "A proposed answer to the following problem is {answer}.\n\n"
    "Problem: {problem}\n\n"
    "Your task: SOLVE this problem from scratch using a COMPLETELY DIFFERENT method "
    "than you would normally use. Use Python to compute. "
    "Compare your independent result with the proposed answer. "
    "After your analysis, you MUST print your conclusion as a Python print statement:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer inside \\boxed{{}}.\n"
    "IMPORTANT: You MUST call print() with the VERDICT line — this is required."
)

VERIFY_COUNTEREXAMPLE = (
    "A proposed answer to the following problem is {answer}.\n\n"
    "Problem: {problem}\n\n"
    "Your task: TRY TO FIND A COUNTEREXAMPLE that shows the proposed answer is wrong. "
    "Use Python to test specific cases, edge cases, or construct a contradiction. "
    "If you find a counterexample, the answer is incorrect. "
    "If you cannot find a counterexample after thorough testing, the answer is likely correct. "
    "After your analysis, you MUST print your conclusion as a Python print statement:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer inside \\boxed{{}}.\n"
    "IMPORTANT: You MUST call print() with the VERDICT line — this is required."
)

# Ordered list — we rotate through them for each candidate's verification attempts.
VERIFY_STRATEGIES = [
    VERIFY_SUBSTITUTION,
    VERIFY_SMALL_CASES,
    VERIFY_ALTERNATIVE,
    VERIFY_COUNTEREXAMPLE,
]


class AIMO3Template:
    """AIMO-3 prompt template management with lazy Harmony imports."""

    def __init__(self):
        self._h = _require_harmony()

    @staticmethod
    def _default_model_identity() -> str:
        # Per the Harmony prompt-format guidance, the system message identity should remain stable.
        return "You are ChatGPT, a large language model trained by OpenAI."

    def get_system_content(self, tool_config):
        SystemContent = self._h["SystemContent"]
        ReasoningEffort = self._h["ReasoningEffort"]
        return (
            SystemContent.new()
            .with_model_identity(self._default_model_identity())
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(self, developer_prompt: str, user_prompt: str, tool_config):
        Message = self._h["Message"]
        Role = self._h["Role"]
        system_content = self.get_system_content(tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)

        # The project prompts ("system prompts" in older code) are the DEVELOPER instructions
        # in Harmony terms.
        developer_content = None
        if "DeveloperContent" in self._h:
            DeveloperContent = self._h["DeveloperContent"]
            with contextlib.suppress(Exception):
                developer_content = DeveloperContent.new().with_instructions(
                    developer_prompt
                )

        if developer_content is not None:
            developer_message = Message.from_role_and_content(
                Role.DEVELOPER, developer_content
            )
        else:
            # Compatibility fallback for older openai_harmony builds.
            developer_message = Message.from_role_and_content(
                Role.DEVELOPER, developer_prompt
            )

        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, developer_message, user_message]
