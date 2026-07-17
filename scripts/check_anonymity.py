#!/usr/bin/env python3
"""Cross-platform scan for common identity and machine-metadata leaks."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "private", "generated"}
SKIP_SUFFIXES = {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".npy", ".npz", ".zip", ".gz", ".pdf"}

PATTERNS = {
    "email address": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "Unix home path": re.compile(r"/(?:home|Users)/[^/\s]+/", re.I),
    "Windows user path": re.compile(r"[A-Z]:\\Users\\[^\\\s]+\\", re.I),
    "GitHub owner URL": re.compile(r"https?://(?:www\.)?github\.com/[^\s/)]+", re.I),
    "40-character source commit": re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", re.I),
    "machine or experiment service marker": re.compile(
        r"(?:ubuntu@|events\.out\.tfevents|wandb(?:\.ai|/)|mlruns/|instance-[a-z0-9_-]+|thunder[-_ ]?compute)",
        re.I,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on any finding")
    return parser.parse_args()


def text_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            # The scanner necessarily contains the literal deny-list patterns.
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in data:
            continue
        try:
            yield path, data.decode("utf-8")
        except UnicodeDecodeError:
            continue


def git_context(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    try:
        inside = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return []
    details: list[str] = []
    for command, label in (
        (["git", "-C", str(root), "remote", "-v"], "Git remotes require manual review"),
        (["git", "-C", str(root), "log", "-5", "--format=%an <%ae>"], "Recent Git identities require manual review"),
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.stdout.strip():
            details.append(f"{label}:\n{result.stdout.strip()}")
    return details


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    findings: list[str] = []
    for path, text in text_files(root):
        relative = path.relative_to(root)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: {label}: {line.strip()[:180]}")

    if findings:
        print("Potential anonymity findings:")
        for finding in findings:
            print(f"- {finding}")
    else:
        print("PASS: no email, personal home path, GitHub-owner URL, source commit, or machine marker found.")

    git_details = git_context(root)
    if git_details:
        print("\nManual Git review:")
        for detail in git_details:
            print(detail)

    if args.strict and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
