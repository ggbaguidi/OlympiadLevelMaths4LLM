from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LeanToolchainInfo:
    enabled: bool
    installed: bool
    archive_path: str | None
    extracted_root: str | None
    bin_dir: str | None
    lean_path: str | None
    lake_path: str | None
    lean_version: str | None
    lake_version: str | None


@dataclass(frozen=True)
class LeanSmokeTestResult:
    ok: bool
    lean_path: str | None
    lake_path: str | None
    lean_version: str | None
    lake_version: str | None
    file_path: str | None
    exit_code: int | None
    stdout: str
    stderr: str
    elapsed_s: float


_LEAN_CALL_RE = re.compile(
    r"(?is)"  # ignorecase + dotall
    r"(" 
    r"subprocess\.(run|call|check_output|check_call|Popen)"  # common subprocess entrypoints
    r"|os\.system"
    r")"
)

_LEAN_WORD_RE = re.compile(r"(?i)\b(lean|lake)\b")


def detect_lean_invocation(python_code: str | None) -> bool:
    """Best-effort heuristic: does this python tool call likely invoke lean/lake?"""

    s = (python_code or "")
    if not s.strip():
        return False
    if _LEAN_WORD_RE.search(s) is None:
        return False
    # Require some indication it's a shell/subprocess invocation.
    if _LEAN_CALL_RE.search(s) is None and "!lean" not in s and "!lake" not in s:
        return False
    return True


def _is_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() not in {"", "0", "false", "no"}


def _find_executable(name: str) -> str | None:
    p = shutil.which(name)
    if p:
        return p

    # Common elan location
    elan_bin = Path.home() / ".elan" / "bin" / name
    if elan_bin.exists():
        return str(elan_bin)

    return None


def _safe_run_version(cmd: list[str]) -> str | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except Exception:
        return None
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    txt = out if out else err
    return txt.strip() if txt else None


def _default_smoke_dir() -> Path:
    kaggle_work = Path("/kaggle/working")
    if kaggle_work.exists():
        return kaggle_work / "lean4_smoke"
    return Path(".cache") / "aimo3" / "lean4_smoke"


def lean_smoke_test(
    *,
    work_dir: str | None = None,
    text: str | None = None,
    timeout_s: float = 15.0,
    verbose: bool = False,
) -> LeanSmokeTestResult:
    """Typecheck a tiny Lean file to confirm the toolchain works.

    Intended usage in a Kaggle submission notebook BEFORE solver startup:
    1) ensure_lean_toolchain(...)
    2) lean_smoke_test(...)

    Returns a structured result (stdout/stderr/exit code) for easy debugging.
    """

    lean_path = _find_executable("lean")
    lake_path = _find_executable("lake")
    lean_ver = _safe_run_version(["lean", "--version"]) if lean_path else None
    lake_ver = _safe_run_version(["lake", "--version"]) if lake_path else None

    if not lean_path:
        raise FileNotFoundError("`lean` not found on PATH. Did you run ensure_lean_toolchain()?")

    smoke_root = Path(work_dir) if work_dir else _default_smoke_dir()
    smoke_root = smoke_root.expanduser().resolve()
    smoke_root.mkdir(parents=True, exist_ok=True)

    if text is None:
        # Keep it minimal: no Mathlib, only core.
        text = """-- aimo3 lean smoke test\n#eval (1 + 1 : Nat)\nexample : (1 : Nat) = 1 := rfl\n"""

    test_file = smoke_root / "AIMO3Smoke.lean"
    test_file.write_text(text, encoding="utf-8")

    start = time.time()
    try:
        p = subprocess.run(
            ["lean", str(test_file)],
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
        exit_code = int(p.returncode)
        stdout = (p.stdout or "")
        stderr = (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        exit_code = None
        stdout = (getattr(e, "stdout", None) or "")
        stderr = (getattr(e, "stderr", None) or "")
        stderr = (stderr + "\n" if stderr else "") + f"[ERROR] lean smoke test timed out after {timeout_s} seconds"
    except Exception as e:  # noqa: BLE001
        exit_code = None
        stdout = ""
        stderr = f"[ERROR] lean smoke test failed: {e}"

    elapsed = time.time() - start
    ok = exit_code == 0

    if verbose:
        print(f"[lean-smoke] lean: {lean_path}")
        if lake_path:
            print(f"[lean-smoke] lake: {lake_path}")
        if lean_ver:
            print(f"[lean-smoke] lean --version: {lean_ver}")
        if lake_ver:
            print(f"[lean-smoke] lake --version: {lake_ver}")
        print(f"[lean-smoke] file: {test_file}")
        print(f"[lean-smoke] exit: {exit_code}  elapsed_s={elapsed:.3f}")
        if stdout.strip():
            print("[lean-smoke] stdout:\n" + stdout.strip())
        if stderr.strip():
            print("[lean-smoke] stderr:\n" + stderr.strip())

    return LeanSmokeTestResult(
        ok=ok,
        lean_path=lean_path,
        lake_path=lake_path,
        lean_version=lean_ver,
        lake_version=lake_ver,
        file_path=str(test_file),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        elapsed_s=float(elapsed),
    )


def _default_work_dir() -> Path:
    # Kaggle: /kaggle/working is writable.
    kaggle_work = Path("/kaggle/working")
    if kaggle_work.exists():
        return kaggle_work / "lean4"

    # Local dev: keep it in a project-local cache.
    return Path(".cache") / "aimo3" / "lean4"


def _locate_archive(dataset_dir: Path, archive_name: str | None = None) -> Path | None:
    if archive_name:
        p = dataset_dir / archive_name
        return p if p.exists() else None

    # Prefer the conventional naming.
    cands = sorted(dataset_dir.glob("lean-*-linux.tar.gz"))
    if cands:
        return cands[0]

    cands = sorted(dataset_dir.glob("lean-*-linux.tar"))
    if cands:
        return cands[0]

    # Fallback to any tar-like archive.
    cands = sorted(dataset_dir.glob("*.tar.gz"))
    if cands:
        return cands[0]

    cands = sorted(dataset_dir.glob("*.tar"))
    if cands:
        return cands[0]

    return None


def _find_bin_dir(root: Path) -> Path | None:
    # Search for bin/lean; the release usually has a nested top-level folder.
    hits = sorted(root.rglob("bin/lean"))
    if not hits:
        return None
    return hits[0].parent


def ensure_lean_toolchain(
    *,
    enabled: bool | None = None,
    dataset_dir: str | None = None,
    archive_path: str | None = None,
    archive_name: str | None = None,
    work_dir: str | None = None,
    prefer_existing: bool = True,
    check_versions: bool = True,
    verbose: bool = False,
    strict: bool = False,
) -> LeanToolchainInfo:
    """Ensure Lean4 (`lean`, `lake`) is available on PATH.

    This is intended for Kaggle offline runtimes:
    - You upload a prepacked `lean-<ver>-linux.tar.gz` as a Kaggle Dataset.
    - At runtime we extract it to a writable directory and prepend its `bin/` to PATH.

    The function is idempotent and safe to call multiple times.

    Environment variables (used as defaults when corresponding args are None):
    - AIMO3_LEAN_TOOLCHAIN_ENABLED
    - AIMO3_LEAN_DATASET_DIR
    - AIMO3_LEAN_ARCHIVE_PATH
    - AIMO3_LEAN_ARCHIVE_NAME
    - AIMO3_LEAN_WORK_DIR

    Side-effect:
    - sets AIMO3_LEAN_BIN_DIR when it finds/extracts a toolchain
    - prepends that directory to PATH in the current process
    """

    if enabled is None:
        enabled = _is_truthy(os.getenv("AIMO3_LEAN_TOOLCHAIN_ENABLED", "0"))
    enabled = bool(enabled)

    if not enabled:
        return LeanToolchainInfo(
            enabled=False,
            installed=False,
            archive_path=None,
            extracted_root=None,
            bin_dir=None,
            lean_path=_find_executable("lean"),
            lake_path=_find_executable("lake"),
            lean_version=None,
            lake_version=None,
        )

    # Fast path: already installed.
    lean_path = _find_executable("lean")
    lake_path = _find_executable("lake")

    if prefer_existing and lean_path and lake_path:
        lean_ver = _safe_run_version(["lean", "--version"]) if check_versions else None
        lake_ver = _safe_run_version(["lake", "--version"]) if check_versions else None
        return LeanToolchainInfo(
            enabled=True,
            installed=True,
            archive_path=None,
            extracted_root=None,
            bin_dir=None,
            lean_path=lean_path,
            lake_path=lake_path,
            lean_version=lean_ver,
            lake_version=lake_ver,
        )

    # If a previous call already discovered the bin dir, reuse it.
    existing_bin = os.getenv("AIMO3_LEAN_BIN_DIR")
    if existing_bin and Path(existing_bin).exists():
        os.environ["PATH"] = str(existing_bin) + os.pathsep + os.environ.get("PATH", "")
        lean_path = _find_executable("lean")
        lake_path = _find_executable("lake")
        if lean_path and lake_path:
            lean_ver = _safe_run_version(["lean", "--version"]) if check_versions else None
            lake_ver = _safe_run_version(["lake", "--version"]) if check_versions else None
            return LeanToolchainInfo(
                enabled=True,
                installed=True,
                archive_path=None,
                extracted_root=None,
                bin_dir=str(existing_bin),
                lean_path=lean_path,
                lake_path=lake_path,
                lean_version=lean_ver,
                lake_version=lake_ver,
            )

    if dataset_dir is None:
        dataset_dir = os.getenv("AIMO3_LEAN_DATASET_DIR")
    if archive_path is None:
        archive_path = os.getenv("AIMO3_LEAN_ARCHIVE_PATH")
    if archive_name is None:
        archive_name = os.getenv("AIMO3_LEAN_ARCHIVE_NAME")

    if work_dir is None:
        work_dir = os.getenv("AIMO3_LEAN_WORK_DIR")
    root = Path(work_dir) if work_dir else _default_work_dir()
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    archive: Path | None = None

    # If the user passes a directory as archive_path, treat it as an already-extracted root.
    extracted_hint: Path | None = None
    if archive_path:
        ap = Path(archive_path).expanduser()
        if ap.exists() and ap.is_dir():
            extracted_hint = ap

    if archive_path:
        p = Path(archive_path).expanduser()
        archive = p if p.exists() and p.is_file() else None
    if archive is None and dataset_dir:
        archive = _locate_archive(Path(dataset_dir).expanduser(), archive_name=archive_name)

    if archive is None:
        # Some workflows place an *already extracted* Lean distribution in dataset_dir.
        # If so, accept it and just add its bin dir to PATH.
        for candidate_root in [extracted_hint, Path(dataset_dir).expanduser() if dataset_dir else None]:
            if candidate_root is None:
                continue
            if candidate_root.exists() and candidate_root.is_dir():
                bin_dir0 = _find_bin_dir(candidate_root)
                if bin_dir0 is not None:
                    os.environ["AIMO3_LEAN_BIN_DIR"] = str(bin_dir0)
                    os.environ["PATH"] = str(bin_dir0) + os.pathsep + os.environ.get("PATH", "")

                    lean_path = _find_executable("lean")
                    lake_path = _find_executable("lake")
                    lean_ver = _safe_run_version(["lean", "--version"]) if check_versions else None
                    lake_ver = _safe_run_version(["lake", "--version"]) if check_versions else None

                    installed0 = bool(lean_path and lake_path)
                    if strict and not installed0:
                        raise RuntimeError(
                            "Found an extracted Lean bin dir, but lean/lake still not found on PATH"
                        )

                    return LeanToolchainInfo(
                        enabled=True,
                        installed=installed0,
                        archive_path=None,
                        extracted_root=str(candidate_root),
                        bin_dir=str(bin_dir0),
                        lean_path=lean_path,
                        lake_path=lake_path,
                        lean_version=lean_ver,
                        lake_version=lake_ver,
                    )

        msg = (
            "Lean toolchain is enabled but no archive was found. "
            "Set AIMO3_LEAN_DATASET_DIR to your Kaggle dataset mount (e.g. /kaggle/input/<ds>) "
            "or set AIMO3_LEAN_ARCHIVE_PATH to the full path of lean-<ver>-linux.tar.gz."
        )
        if strict:
            raise FileNotFoundError(msg)
        if verbose:
            print(msg)
        return LeanToolchainInfo(
            enabled=True,
            installed=False,
            archive_path=None,
            extracted_root=str(root),
            bin_dir=None,
            lean_path=None,
            lake_path=None,
            lean_version=None,
            lake_version=None,
        )

    if verbose:
        print(f"[lean] Using archive: {archive}")
        print(f"[lean] Extract root: {root}")

    # Extract (idempotent-ish): if we already have a bin dir, don't re-extract.
    bin_dir = _find_bin_dir(root)
    if bin_dir is None:
        # Use auto-detection so we can handle .tar.gz and .tar.
        with tarfile.open(archive, "r:*") as tf:
            # Python 3.14+ changes default extraction safety behavior.
            # Use the new filter when available to avoid warnings and keep behavior explicit.
            try:
                tf.extractall(path=root, filter="data")
            except TypeError:
                tf.extractall(path=root)
        bin_dir = _find_bin_dir(root)

    if bin_dir is None:
        msg = f"Archive extracted but bin/lean not found under {root}" \
            f" (archive={archive})."
        if strict:
            raise RuntimeError(msg)
        if verbose:
            print("[lean] " + msg)
        return LeanToolchainInfo(
            enabled=True,
            installed=False,
            archive_path=str(archive),
            extracted_root=str(root),
            bin_dir=None,
            lean_path=None,
            lake_path=None,
            lean_version=None,
            lake_version=None,
        )

    # Make available in the parent process.
    os.environ["AIMO3_LEAN_BIN_DIR"] = str(bin_dir)
    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

    lean_path = _find_executable("lean")
    lake_path = _find_executable("lake")

    lean_ver = _safe_run_version(["lean", "--version"]) if check_versions else None
    lake_ver = _safe_run_version(["lake", "--version"]) if check_versions else None

    installed = bool(lean_path and lake_path)

    if verbose:
        print(f"[lean] Installed: {installed}")
        if lean_ver:
            print(f"[lean] lean --version: {lean_ver}")
        if lake_ver:
            print(f"[lean] lake --version: {lake_ver}")

    if strict and not installed:
        raise RuntimeError("Lean toolchain setup completed, but lean/lake still not found on PATH")

    return LeanToolchainInfo(
        enabled=True,
        installed=installed,
        archive_path=str(archive),
        extracted_root=str(root),
        bin_dir=str(bin_dir),
        lean_path=lean_path,
        lake_path=lake_path,
        lean_version=lean_ver,
        lake_version=lake_ver,
    )
