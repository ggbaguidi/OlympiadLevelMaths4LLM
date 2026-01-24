"""AIMO-3 (AI Mathematical Olympiad) runner components.

This subpackage refactors the provided notebook-style solution (`aimo-3.py`) into
importable modules.

Design goals:
- keep optional, heavy dependencies (vLLM/OpenAI Harmony/Kaggle) out of import-time
  paths so the core project remains lightweight
- expose clean building blocks: config, prompts, sandboxed python tool, solver loop

If you want a ready-to-run entrypoint, see `olympiad_llm.aimo3.runner`.
"""

from .errors import OptionalDependencyError
from .config import AIMO3Config

__all__ = ["AIMO3Config", "OptionalDependencyError"]
