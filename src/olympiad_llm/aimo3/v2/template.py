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
    "Proposed answer: {answer}\n"
    "Problem: {problem}\n\n"
    "Check it by direct substitution in Python. Compute explicitly; do not skip steps.\n"
    "At the end print exactly one of:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer as \\boxed{{n}}."
)

VERIFY_SMALL_CASES = (
    "Proposed answer: {answer}\n"
    "Problem: {problem}\n\n"
    "Check it with small cases and boundary cases in Python. "
    "Rebuild the answer from scratch on tiny instances, then compare.\n"
    "At the end print exactly one of:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer as \\boxed{{n}}."
)

VERIFY_ALTERNATIVE = (
    "Proposed answer: {answer}\n"
    "Problem: {problem}\n\n"
    "Solve it again with a different exact method. Use Python to compute and compare.\n"
    "At the end print exactly one of:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer as \\boxed{{n}}."
)

VERIFY_COUNTEREXAMPLE = (
    "Proposed answer: {answer}\n"
    "Problem: {problem}\n\n"
    "Try to find a counterexample in Python. Test sharp cases, edge cases, or contradictions.\n"
    "At the end print exactly one of:\n"
    '  print("VERDICT: CORRECT")\n'
    '  print("VERDICT: INCORRECT")\n'
    "If incorrect, also print the correct answer as \\boxed{{n}}."
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

    def get_system_content(self, tool_configs):
        SystemContent = self._h["SystemContent"]
        ReasoningEffort = self._h["ReasoningEffort"]
        return (
            SystemContent.new()
            .with_model_identity(self._default_model_identity())
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_configs)
        )

    def apply_chat_template(
        self, developer_prompt: str, user_prompt: str, tool_configs
    ):
        Message = self._h["Message"]
        Role = self._h["Role"]
        system_content = self.get_system_content(tool_configs)
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
