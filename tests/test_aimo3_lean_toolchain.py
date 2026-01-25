from __future__ import annotations

import os
import stat
import tarfile
from pathlib import Path

import pytest

from olympiad_llm.aimo3.lean_toolchain import ensure_lean_toolchain


def _make_dummy_lean_archive(tmp_path: Path) -> Path:
    """Create a tar.gz that contains a dummy bin/lean and bin/lake."""
    # Mimic the nested folder layout seen in Lean releases.
    root = tmp_path / "lean-4.14.0-linux"
    nested = root / "lean-4.14.0-linux" / "bin"
    nested.mkdir(parents=True, exist_ok=True)

    lean = nested / "lean"
    lake = nested / "lake"

    for p in (lean, lake):
        p.write_text("#!/bin/sh\necho dummy\n", encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    out = tmp_path / "lean-4.14.0-linux.tar.gz"
    with tarfile.open(out, "w:gz") as tf:
        tf.add(root, arcname=root.name)

    return out


def _make_dummy_lean_tar(tmp_path: Path) -> Path:
    """Create an uncompressed .tar that contains a dummy bin/lean and bin/lake."""
    root = tmp_path / "lean-4.14.0-linux"
    nested = root / "lean-4.14.0-linux" / "bin"
    nested.mkdir(parents=True, exist_ok=True)

    lean = nested / "lean"
    lake = nested / "lake"

    for p in (lean, lake):
        p.write_text("#!/bin/sh\necho dummy\n", encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    out = tmp_path / "lean-4.14.0-linux.tar"
    with tarfile.open(out, "w") as tf:
        tf.add(root, arcname=root.name)
    return out


def test_ensure_lean_toolchain_extracts_and_sets_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = _make_dummy_lean_archive(tmp_path)

    # Ensure we don't accidentally pick up a real lean/lake from the host.
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("AIMO3_LEAN_BIN_DIR", raising=False)

    info = ensure_lean_toolchain(
        enabled=True,
        archive_path=str(archive),
        work_dir=str(tmp_path / "work"),
        prefer_existing=False,
        check_versions=False,
        verbose=False,
        strict=True,
    )

    assert info.enabled is True
    assert info.installed is True
    assert info.archive_path is not None
    assert info.bin_dir is not None
    assert (Path(info.bin_dir) / "lean").exists()
    assert os.environ.get("AIMO3_LEAN_BIN_DIR") == info.bin_dir
    assert os.environ.get("PATH", "").startswith(info.bin_dir)


def test_ensure_lean_toolchain_accepts_tar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    archive = _make_dummy_lean_tar(tmp_path)

    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("AIMO3_LEAN_BIN_DIR", raising=False)

    info = ensure_lean_toolchain(
        enabled=True,
        archive_path=str(archive),
        work_dir=str(tmp_path / "work"),
        prefer_existing=False,
        check_versions=False,
        verbose=False,
        strict=True,
    )

    assert info.enabled is True
    assert info.installed is True
    assert info.bin_dir is not None
    assert os.environ.get("PATH", "").startswith(info.bin_dir)


def test_ensure_lean_toolchain_missing_archive_non_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("AIMO3_LEAN_BIN_DIR", raising=False)

    info = ensure_lean_toolchain(
        enabled=True,
        dataset_dir=str(tmp_path / "does-not-exist"),
        work_dir=str(tmp_path / "work"),
        prefer_existing=False,
        check_versions=False,
        verbose=False,
        strict=False,
    )

    assert info.enabled is True
    assert info.installed is False


def test_ensure_lean_toolchain_missing_archive_strict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("AIMO3_LEAN_BIN_DIR", raising=False)

    with pytest.raises(FileNotFoundError):
        ensure_lean_toolchain(
            enabled=True,
            dataset_dir=str(tmp_path / "does-not-exist"),
            work_dir=str(tmp_path / "work"),
            prefer_existing=False,
            check_versions=False,
            verbose=False,
            strict=True,
        )


def test_ensure_lean_toolchain_accepts_extracted_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Create an extracted-like layout (no tar.gz present)
    dataset = tmp_path / "lean4_offline_bundle"
    bin_dir = dataset / "lean-4.14.0-linux" / "lean-4.14.0-linux" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    lean = bin_dir / "lean"
    lake = bin_dir / "lake"
    for p in (lean, lake):
        p.write_text("#!/bin/sh\necho dummy\n", encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("AIMO3_LEAN_BIN_DIR", raising=False)

    info = ensure_lean_toolchain(
        enabled=True,
        dataset_dir=str(dataset),
        prefer_existing=False,
        check_versions=False,
        verbose=False,
        strict=True,
    )

    assert info.enabled is True
    assert info.installed is True
    assert info.bin_dir is not None
    assert Path(info.bin_dir) == bin_dir
    assert os.environ.get("PATH", "").startswith(str(bin_dir))
