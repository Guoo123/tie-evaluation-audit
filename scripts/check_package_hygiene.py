#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "ARTIFACT_MANIFEST.sha256"

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".csv", ".cff", ".sh"
}
SKIP_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"
}
SKIP_PREFIXES = (
    Path("data/raw"), Path("data/interim"), Path("data/processed"), Path("results/regenerated")
)
ALLOWED_PLACEHOLDERS = {prefix / ".gitkeep" for prefix in SKIP_PREFIXES}
OS_METADATA_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}
HOME_PATH = re.compile(
    r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)"
)
STALE_PUBLIC_METADATA = (
    "# Anonymous artifact",
    "Anonymous reproducibility artifact",
    "During anonymous review",
)


def _is_skipped(relative: Path) -> bool:
    if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return True
    return any(relative == prefix or prefix in relative.parents for prefix in SKIP_PREFIXES)


def _under_generated_prefix(relative: Path) -> bool:
    return any(relative == prefix or prefix in relative.parents for prefix in SKIP_PREFIXES)


def _manifested_paths(findings: list[str]) -> set[str]:
    if not MANIFEST.is_file():
        findings.append("ARTIFACT_MANIFEST.sha256: missing distribution manifest")
        return set()

    manifested: set[str] = set()
    for line_number, line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            _, relative = line.split("  ", 1)
        except ValueError:
            findings.append(
                f"ARTIFACT_MANIFEST.sha256:{line_number}: malformed manifest row"
            )
            continue
        manifested.add(relative)
    return manifested


def main() -> None:
    findings: list[str] = []
    manifested = _manifested_paths(findings)
    for path in sorted(REPO_ROOT.rglob("*")):
        relative = path.relative_to(REPO_ROOT)
        if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if any(
            part == "__MACOSX"
            or part.startswith("._")
            or part.lower() in OS_METADATA_NAMES
            for part in relative.parts
        ):
            findings.append(f"operating-system metadata: {relative}")
            continue
        if path.is_symlink():
            findings.append(f"symbolic link is not allowed in the release package: {relative}")
            continue
        if not path.is_file():
            continue
        if _under_generated_prefix(relative):
            if relative not in ALLOWED_PLACEHOLDERS:
                findings.append(f"generated or downloaded file must not be packaged: {relative}")
            continue
        if relative != Path("ARTIFACT_MANIFEST.sha256") and relative.as_posix() not in manifested:
            findings.append(f"distributable file is not listed in the manifest: {relative}")
        if _is_skipped(relative):
            continue
        if relative == Path("scripts/check_package_hygiene.py"):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Dockerfile", "Makefile"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in HOME_PATH.finditer(text):
            findings.append(f"{relative}: user-specific home path: {match.group(0)}")

    for relative in (Path("README.md"), Path("pyproject.toml"), Path("CITATION.cff")):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for phrase in STALE_PUBLIC_METADATA:
            if phrase in text:
                findings.append(f"{relative}: stale anonymous-review wording: {phrase}")

    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "authors:" not in citation:
        findings.append("CITATION.cff: missing author metadata")
    if "https://github.com/Guoo123/tie-evaluation-audit" not in citation:
        findings.append("CITATION.cff: missing repository-code URL")

    if findings:
        print("Package hygiene check failed:")
        print("\n".join(findings))
        raise SystemExit(1)
    print(
        "Package hygiene check passed: no local home paths, OS metadata, symbolic links, "
        "unexpected generated files, or unmanifested distributable files were found; "
        "the public citation metadata is present and non-anonymous."
    )


if __name__ == "__main__":
    main()
