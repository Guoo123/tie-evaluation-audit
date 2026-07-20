#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "ARTIFACT_MANIFEST.sha256"
SKIP_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
}
SKIP_PREFIXES = (
    Path("data/raw"),
    Path("data/interim"),
    Path("data/processed"),
    Path("results/regenerated"),
)


def skipped(relative: Path, output: Path) -> bool:
    if relative == output.relative_to(REPO_ROOT):
        return True
    if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return True
    return any(relative == prefix or prefix in relative.parents for prefix in SKIP_PREFIXES)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def build(output: Path = DEFAULT_OUTPUT) -> int:
    rows: list[str] = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if skipped(relative, output):
            continue
        rows.append(f"{digest(path)}  {relative.as_posix()}")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a SHA-256 manifest of distributable files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    count = build(output)
    print(f"Wrote {output.relative_to(REPO_ROOT)} with {count} file hashes.")


if __name__ == "__main__":
    main()
