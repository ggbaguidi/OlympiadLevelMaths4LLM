"""Environment cleanup + offline install helper.

This is a direct port of the notebook's `cleanup.py`, but:
- it is *optional* (not used by default)
- it is safe to import outside Kaggle

In Kaggle, you can use this to install wheels from a mounted dataset.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any


IS_KAGGLE = os.path.exists("/kaggle/input")
PACKAGE_MANAGER = os.getenv("AIMO3_PACKAGE_MANAGER", "pip")
UNINSTALL_PACKAGES_COMMAND = ["uninstall", "--yes"]
CONFLICTING_LIBRARIES = ["tensorflow", "matplotlib", "keras", "scikit-learn"]

REQUIRED_LIBRARIES = [
    "torch",
    "numpy",
    "unsloth",
    "trl",
    "vllm",
    "openai_harmony",
    "mpmath",
    "ortools",
]


def _packages_path() -> str:
    if IS_KAGGLE:
        return os.getenv("AIMO3_WHEELS_PATH", "/kaggle/usr/lib/aimo3_packages_offline/utils")
    return os.getenv("AIMO3_WHEELS_PATH", "")


def uninstall_conflicts() -> Any:
    return subprocess.Popen(
        [PACKAGE_MANAGER, *UNINSTALL_PACKAGES_COMMAND, *CONFLICTING_LIBRARIES],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def install_offline_required() -> Any:
    wheels = _packages_path()
    if not wheels:
        raise RuntimeError(
            "Offline wheels path not set. Set AIMO3_WHEELS_PATH or run in Kaggle where it is mounted."
        )

    return subprocess.Popen(
        [
            PACKAGE_MANAGER,
            "install",
            "--no-index",
            "--find-links=" + wheels,
            *REQUIRED_LIBRARIES,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def environ_setup() -> dict[str, Any]:
    return {
        "uninstall_process": uninstall_conflicts(),
        "install_process": install_offline_required(),
    }
