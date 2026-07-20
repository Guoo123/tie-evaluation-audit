# Anonymization record

The distributable artifact intentionally excludes:

- the manuscript PDF and all author/contact metadata;
- the original repository's Git metadata, remotes, hosted-account URL, and source
  commit identifiers;
- personal email addresses and user-specific home paths;
- internal planning notes, broad unrelated experiments, logs, TensorBoard records,
  and machine-specific output bundles; and
- the original development repository name.

The package retains only the paper title, public dataset names and URLs, scientific
configuration values, and the anonymous artifact's own prospective runtime metadata.
A reviewer checkout may record its own commit hash in a newly generated
`runtime_manifest.json`; generated outputs are not part of the distributed package.

Run:

```bash
python scripts/check_anonymity.py
```

The check scans distributable text files and fails on manuscript-like files, email
addresses, user home paths, hosted source-control account URLs, and retained source-
repository commit fields. Before a public release after acceptance, replace the
anonymous citation metadata and choose a final source-code license.
