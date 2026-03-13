# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name

from __future__ import annotations

import contextlib
import importlib

from .errors import OptionalDependencyError


def _require_openai():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            "AIMO3Solver requires 'openai'. Install extras: pip install .[aimo3]"
        ) from e
    return OpenAI


def _require_harmony():
    try:
        from openai_harmony import (
            Author,
            Conversation,  # type: ignore
            HarmonyEncodingName,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
            TextContent,
            ToolNamespaceConfig,
            load_harmony_encoding,
        )
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            "AIMO3Solver requires 'openai_harmony'. Install it from your offline wheels or pip (package may be named openai-harmony)."
        ) from e

    # DeveloperContent was introduced in newer openai_harmony releases.
    DeveloperContent = None
    with contextlib.suppress(Exception):
        from openai_harmony import DeveloperContent as _DeveloperContent  # type: ignore

        DeveloperContent = _DeveloperContent

    return {
        "HarmonyEncodingName": HarmonyEncodingName,
        "load_harmony_encoding": load_harmony_encoding,
        "SystemContent": SystemContent,
        "DeveloperContent": DeveloperContent,
        "ReasoningEffort": ReasoningEffort,
        "ToolNamespaceConfig": ToolNamespaceConfig,
        "Author": Author,
        "Message": Message,
        "Role": Role,
        "TextContent": TextContent,
        "Conversation": Conversation,
    }


def _require_llama_cpp_server() -> None:
    try:
        importlib.import_module("llama_cpp.server")
    except Exception as e:  # noqa: BLE001
        raise OptionalDependencyError(
            "llama.cpp backend requires 'llama-cpp-python[server]'. Install extras: pip install .[llama-cpp]"
        ) from e
