# Package hygiene and release metadata

The focused artifact excludes local machine paths, operating-system metadata,
unrelated experiments, logs, model checkpoints, and raw datasets. These exclusions
keep reviewer inspection focused and avoid accidental disclosure of machine-specific
information.

FRAME Track 2 uses single-blind review. The public repository therefore retains the
paper title, author and citation metadata, contact emails, and the hosted repository
URL. Archived files that describe the earlier sanitized development bundle are kept
only as provenance records.

Run:

```bash
python scripts/check_package_hygiene.py
```

The check fails on `__MACOSX`, AppleDouble `._*`, `.DS_Store`, `Thumbs.db`, or
`desktop.ini`
material; symbolic links; unexpected files under generated/downloaded-data
directories; distributable files missing from `ARTIFACT_MANIFEST.sha256`;
user-specific home paths; stale anonymous-review wording in the public metadata
files; or missing citation metadata.
