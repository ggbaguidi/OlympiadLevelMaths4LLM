# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name
from __future__ import annotations

import contextlib
from .require import _require_harmony


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
