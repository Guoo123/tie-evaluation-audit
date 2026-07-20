# Result workspace

`archived/` contains sanitized aggregate/configuration evidence that was actually
present in the supplied development ZIP. It supports paper-claim and arithmetic
checks but not historical row-level replay.

`regenerated/` is the local output root for new public-data runs. Its contents are
ignored by Git. A complete run retains candidate rows, candidate scores, all tie
policies, diagnostics, environment/configuration records, and SHA-256 manifests so it
can be replayed without preprocessing or resampling.
