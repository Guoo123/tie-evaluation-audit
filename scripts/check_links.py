#!/usr/bin/env python3
"""Fail on broken relative Markdown links in the reviewer artifact."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def main() -> None:
    broken: list[str] = []
    checked = 0
    for markdown in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", ".venv", ".pytest_cache", "private", "generated"} for part in markdown.relative_to(ROOT).parts):
            continue
        text = markdown.read_text(encoding="utf-8")
        for match in LINK.finditer(text):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = unquote(target.split("#", 1)[0])
            if not relative_target:
                continue
            checked += 1
            resolved = (markdown.parent / relative_target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                broken.append(f"{markdown.relative_to(ROOT)} -> {target} (escapes repository)")
                continue
            if not resolved.exists():
                broken.append(f"{markdown.relative_to(ROOT)} -> {target}")

    if broken:
        raise SystemExit("FAIL: broken local Markdown links:\n  " + "\n  ".join(broken))
    print(f"PASS: {checked} relative Markdown links resolve.")


if __name__ == "__main__":
    main()
