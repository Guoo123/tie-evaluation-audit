# Publication and Identity Checklist

This export contains no original Git history and intentionally omits the manuscript PDF, development notes, raw logs, TensorBoard files, and the public development-repository reference.

Before making the repository public for review:

1. Use a clean repository rather than forking or exposing the development repository and its history.
2. Run `python scripts/check_anonymity.py --strict` and inspect every finding.
3. Run `python scripts/build_manifest.py` and commit the resulting manifest with the artifact snapshot.
4. Clone the published repository in a clean directory or private browser session and repeat the quick start.
5. Confirm that the public URL opens without a GitHub login.
6. Replace the manuscript's development-repository reference only after the clean-clone test passes.

Do not upload the supplied PDF under its current filename: its visible first page contains author information, and a later page names the development repository. The PDF itself is outside this artifact.

FRAME's Experimental Methodologies track is single-blind, so reviewer-facing author identity is permitted. A repository hosted under an author's GitHub account, or carrying a named license and commit metadata, is therefore acceptable but is **not** a double-blind anonymous artifact. If a later submission route requires double-blind review, publish a separate identity-sanitized snapshot through a neutral account or anonymous artifact service.
