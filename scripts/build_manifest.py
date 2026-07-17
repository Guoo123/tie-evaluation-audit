#!/usr/bin/env python3
"""Create or verify a deterministic SHA-256 artifact manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
SUMS = ROOT / "SHA256SUMS.txt"
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "private", "generated"}
SKIP_NAMES = {MANIFEST.name, SUMS.name}
SKIP_SUFFIXES = {".pyc", ".zip", ".gz"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_files() -> dict[str, dict[str, int | str]]:
    files: dict[str, dict[str, int | str]] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.name in SKIP_NAMES or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        key = relative.as_posix()
        files[key] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return files


def write_manifest(files: dict[str, dict[str, int | str]]) -> None:
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "files": files,
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"{metadata['sha256']}  {path}" for path, metadata in sorted(files.items())]
    SUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_manifest(files: dict[str, dict[str, int | str]]) -> None:
    if not MANIFEST.is_file() or not SUMS.is_file():
        raise SystemExit("FAIL: MANIFEST.json or SHA256SUMS.txt is missing; run without --check first")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("FAIL: MANIFEST.json must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise SystemExit("FAIL: unsupported or missing MANIFEST.json schema_version")
    if payload.get("hash_algorithm") != "sha256":
        raise SystemExit("FAIL: MANIFEST.json hash_algorithm must be sha256")
    if payload.get("file_count") != len(files):
        raise SystemExit(
            "FAIL: MANIFEST.json file_count does not match the current artifact "
            f"({payload.get('file_count')!r} != {len(files)})"
        )
    expected = payload.get("files")
    if not isinstance(expected, dict):
        raise SystemExit("FAIL: MANIFEST.json files must be a JSON object")
    if expected != files:
        expected_paths = set(expected or {})
        actual_paths = set(files)
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        changed = sorted(
            path for path in expected_paths & actual_paths if expected[path] != files[path]
        )
        raise SystemExit(
            "FAIL: manifest mismatch\n"
            f"  missing: {missing}\n"
            f"  extra: {extra}\n"
            f"  changed: {changed}"
        )
    expected_sums = "\n".join(
        f"{metadata['sha256']}  {path}" for path, metadata in sorted(files.items())
    ) + "\n"
    if SUMS.read_text(encoding="utf-8") != expected_sums:
        raise SystemExit("FAIL: SHA256SUMS.txt does not match MANIFEST.json")
    print(f"PASS: {len(files)} files match MANIFEST.json and SHA256SUMS.txt.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = current_files()
    if args.check:
        check_manifest(files)
    else:
        write_manifest(files)
        print(f"Wrote {MANIFEST.name} and {SUMS.name} for {len(files)} files.")


if __name__ == "__main__":
    main()
