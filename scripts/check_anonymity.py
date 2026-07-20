#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".csv",
    ".cff",
    ".sh",
}
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
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
HOME_PATH = re.compile(
    r"(?:/" + r"home/[^/\s]+|/" + r"Users/[^/\s]+|[A-Za-z]:\\" + r"Users\\[^\\\s]+)"
)
HOSTED_ACCOUNT = re.compile(
    r"https?://(?:www\.)?(?:git" + r"hub|gitlab|bitbucket)\.com/[^/\s]+",
    re.I,
)
SOURCE_COMMIT = re.compile(
    r'"(?:git_' + r'commit|source_commit|development_repository_snapshot)"\s*:\s*"[0-9a-f]{7,40}"',
    re.I,
)


def _is_skipped(relative: Path) -> bool:
    if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return True
    return any(relative == prefix or prefix in relative.parents for prefix in SKIP_PREFIXES)


def main() -> None:
    findings: list[str] = []
    forbidden_suffixes = {".pdf", ".doc", ".docx"}
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if _is_skipped(relative):
            continue
        if path.suffix.lower() in forbidden_suffixes:
            findings.append(f"{relative}: manuscript-like file must not be in anonymous code package")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Dockerfile", "Makefile"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in (
            ("email", EMAIL),
            ("home path", HOME_PATH),
            ("hosted account", HOSTED_ACCOUNT),
            ("source commit", SOURCE_COMMIT),
        ):
            for match in pattern.finditer(text):
                findings.append(f"{relative}: {label}: {match.group(0)}")
    if findings:
        print("Anonymity check failed:")
        print("\n".join(findings))
        raise SystemExit(1)
    print(
        "Anonymity check passed: tracked artifact files contain no manuscript, email "
        "address, user home path, hosted account URL, or source-repository commit identifier."
    )


if __name__ == "__main__":
    main()
