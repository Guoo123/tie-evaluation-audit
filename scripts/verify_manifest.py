#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "ARTIFACT_MANIFEST.sha256"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify(manifest: Path = DEFAULT_MANIFEST) -> int:
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    checked = 0
    errors: list[str] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"line {line_number}: malformed manifest row")
            continue
        path = REPO_ROOT / relative
        if not path.exists():
            errors.append(f"missing: {relative}")
            continue
        actual = digest(path)
        if actual != expected:
            errors.append(f"checksum mismatch: {relative}")
        checked += 1
    if errors:
        raise AssertionError("\n".join(errors))
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the distributable SHA-256 manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    count = verify(Path(args.manifest).resolve())
    print(f"Artifact manifest verified: {count} files.")


if __name__ == "__main__":
    main()
