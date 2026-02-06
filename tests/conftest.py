"""Pytest configuration.

This repo uses a src-layout (package code under ./src).

In some environments, an older *installed* copy of `olympiad_llm` (or a previous
build artifact) can appear on sys.path ahead of the local workspace code, causing
tests to exercise stale modules.

Force imports to resolve to the local ./src tree.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    build_lib = root / "build" / "lib"

    # Prefer local sources.
    src_str = str(src)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    # Avoid accidentally importing stale build artifacts if present.
    build_str = str(build_lib)
    while build_str in sys.path:
        sys.path.remove(build_str)
