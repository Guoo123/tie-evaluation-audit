from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


def digest(path: str | Path, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _portable_path(path: Path, root: Path | None = None) -> str:
    path = path.resolve()
    if root is not None:
        try:
            return path.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return path.name


def file_record(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    file_path = Path(path)
    return {
        "path": _portable_path(file_path, Path(root) if root is not None else None),
        "bytes": file_path.stat().st_size,
        "sha256": digest(file_path, "sha256"),
    }


def package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_commit(repo_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def runtime_manifest(repo_root: str | Path, config_path: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "git_commit": git_commit(root),
        "config": file_record(config_path, root=root),
        "packages": package_versions(
            [
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "pyarrow",
                "duckdb",
                "PyYAML",
                "requests",
                "tqdm",
                "orjson",
                "matplotlib",
            ]
        ),
    }


def sanitized_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable config without loader-only absolute paths."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): clean(item)
                for key, item in value.items()
                if not str(key).startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, Path):
            return value.as_posix()
        return value

    return clean(config)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def write_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    write_json(sanitized_config(config), path)


def write_run_integrity_manifest(
    output_dir: str | Path,
    repo_root: str | Path,
    *,
    input_paths: Iterable[str | Path] = (),
    filename: str = "run_integrity_manifest.json",
) -> dict[str, Any]:
    """Hash retained run inputs and outputs for prospective row-level replay.

    The manifest deliberately excludes itself and nested replay directories. A replay
    command writes its own input/output manifest.
    """
    output = Path(output_dir)
    root = Path(repo_root)
    destination = output / filename
    inputs = []
    for value in input_paths:
        path = Path(value)
        if path.exists() and path.is_file():
            inputs.append(file_record(path, root=root))

    outputs = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == destination or "replay" in path.relative_to(output).parts:
            continue
        outputs.append(file_record(path, root=root))

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
    }
    write_json(manifest, destination)
    return manifest
