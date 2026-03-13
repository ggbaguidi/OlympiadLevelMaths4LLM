# pylint: disable=broad-exception-caught,missing-function-docstring,line-too-long,missing-module-docstring,import-outside-toplevel,invalid-name,too-many-instance-attributes
"""Environment cleanup + offline install helper.

This is a direct port of the notebook's `cleanup.py`, but:
- it is *optional* (not used by default)
- it is safe to import outside Kaggle

In Kaggle, you can use this to install wheels from a mounted dataset.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

IS_KAGGLE = os.path.exists("/kaggle/input")
PACKAGE_MANAGER = os.getenv("AIMO3_PACKAGE_MANAGER", "pip")
UNINSTALL_PACKAGES_COMMAND = ["uninstall", "--yes"]
# Note: scikit-learn was removed from conflicts because sentence-transformers requires it
CONFLICTING_LIBRARIES = ["tensorflow", "matplotlib", "keras"]

BASE_REQUIRED_LIBRARIES = [
    "torch",
    "numpy",
    "openai",
    "openai-harmony",
    "jupyter_client",
    "ipykernel",
    "pandas",
    "polars",
    "transformers",
    "python-dotenv",
    "sympy",
    "typer",
    "rich",
    "mpmath",
    "ortools",
    "sentence-transformers",
    "scikit-learn",  # Required by sentence-transformers
    "z3-solver",
]

VLLM_REQUIRED_LIBRARIES = [
    "unsloth",
    "trl",
    "vllm",
]

LLAMA_CPP_REQUIRED_LIBRARIES = [
    "llama-cpp-python",
    # Keep server deps explicit for offline installs where extras metadata may
    # not resolve from local wheels.
    "fastapi",
    "uvicorn",
    "sse-starlette",
    "starlette",
    "pydantic",
    "pydantic-settings",
]


def _packages_path() -> str:
    if IS_KAGGLE:
        return os.getenv(
            "AIMO3_WHEELS_PATH", "/kaggle/usr/lib/aimo3_packages_offline/utils"
        )
    return os.getenv("AIMO3_WHEELS_PATH", "")


def uninstall_conflicts() -> Any:
    return subprocess.Popen(
        [PACKAGE_MANAGER, *UNINSTALL_PACKAGES_COMMAND, *CONFLICTING_LIBRARIES],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def resolve_required_libraries() -> list[str]:
    packages = list(BASE_REQUIRED_LIBRARIES)

    backend = (os.getenv("AIMO3_INFERENCE_BACKEND", "vllm") or "vllm").strip().lower()
    offline_stack = (os.getenv("AIMO3_OFFLINE_STACK", "") or "").strip().lower()

    # Install backend-specific runtime dependencies.
    if backend == "vllm":
        packages.extend(VLLM_REQUIRED_LIBRARIES)
    if backend == "llama_cpp":
        packages.extend(LLAMA_CPP_REQUIRED_LIBRARIES)

    # Optional explicit stack selector used by the offline preparation notebook.
    if offline_stack == "full":
        packages.extend(VLLM_REQUIRED_LIBRARIES)

    extra_raw = (os.getenv("AIMO3_EXTRA_REQUIRED_LIBRARIES", "") or "").strip()
    if extra_raw:
        for pkg in extra_raw.split(","):
            name = pkg.strip()
            if name and name not in packages:
                packages.append(name)

    seen: set[str] = set()
    deduped: list[str] = []
    for pkg in packages:
        if pkg not in seen:
            deduped.append(pkg)
            seen.add(pkg)
    return deduped


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
            *resolve_required_libraries(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def warm_model_cache(model_path: str | None = None, workers: int = 8) -> None:
    """Best-effort OS page-cache warmup for model shards (parallel).

    This can run DURING pip install to overlap I/O and save ~30-60s on cold starts.
    """
    model_path = model_path or os.getenv("AIMO3_MODEL_PATH", "")
    if not model_path or not os.path.isdir(model_path):
        return

    # Enumerate shard files
    files: list[str] = []
    for root, _dirs, names in os.walk(model_path):
        for name in names:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                files.append(p)
    if not files:
        return

    workers = max(1, min(workers, os.cpu_count() or 8))

    def _read_file(path: str) -> None:
        try:
            with open(path, "rb") as f:
                while f.read(1024 * 1024 * 1024):  # 1GB chunks
                    pass
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_read_file, files))


def environ_setup() -> dict[str, Any]:
    return {
        "uninstall_process": uninstall_conflicts(),
        "install_process": install_offline_required(),
    }


def environ_setup_parallel(
    warm_model: bool = True, model_workers: int = 8
) -> dict[str, Any]:
    """Run cleanup, install, and model warmup all in parallel.

    Returns dict with process handles. Call wait_all() or .wait() on each to block.
    """
    result: dict[str, Any] = {}

    # Start uninstall
    result["uninstall_process"] = uninstall_conflicts()

    # Start install
    result["install_process"] = install_offline_required()

    # Start model warmup in background thread (overlaps with pip)
    if warm_model:
        model_path = os.getenv("AIMO3_MODEL_PATH", "")
        if model_path and os.path.isdir(model_path):
            ex = ThreadPoolExecutor(max_workers=1)
            result["model_warmup_future"] = ex.submit(
                warm_model_cache, model_path, model_workers
            )
            result["_model_warmup_executor"] = ex  # keep ref to avoid GC

    return result


def wait_all(setup_result: dict[str, Any], timeout: float | None = None) -> None:
    """Wait for all parallel setup tasks to complete."""
    uninstall_rc = None
    install_rc = None

    if "uninstall_process" in setup_result:
        uninstall_rc = setup_result["uninstall_process"].wait(timeout=timeout)
    if "install_process" in setup_result:
        install_rc = setup_result["install_process"].wait(timeout=timeout)
    if "model_warmup_future" in setup_result:
        setup_result["model_warmup_future"].result(timeout=timeout)
    if "_model_warmup_executor" in setup_result:
        setup_result["_model_warmup_executor"].shutdown(wait=False)

    if uninstall_rc not in (None, 0):
        raise RuntimeError(
            f"Conflicting package uninstall step failed with exit code {uninstall_rc}."
        )

    if install_rc not in (None, 0):
        wheels = _packages_path() or "<unset>"
        raise RuntimeError(
            "Offline dependency installation failed "
            f"(exit code {install_rc}). Check your wheels dataset at {wheels} and ensure required packages are present, especially openai-harmony/openai_harmony."
        )
