# Implementation audit of the supplied development repository

The supplied development ZIP was a much broader research repository. It contained
many experiments unrelated to the submitted tie-handling paper, large logs, hardcoded
machine paths, and partial result bundles. Copying it wholesale would have made the
artifact harder to audit and less anonymous. This focused repository was therefore
constructed from the paper-relevant implementation and evidence.

## Components retained or translated

- the 64-bit mixer and historical float32 conversion;
- positive-first candidate construction;
- BPC seed 42 and key seed 20260316;
- MovieLens seed 42 and key seed 20260318;
- BPC lexicons and masked-text propensity hyperparameters;
- BPC sequential iterative 3-core, sorted ID mapping, and per-user temporal split;
- BPC historical rejection sampler;
- MovieLens six tags, 0.7 threshold, positive-rating filter, first-appearance maps,
  global split, first 10,000 test users, and set-based negative sampler;
- archived aggregate evidence and original BPC resolved configuration; and
- the archived BPC `po_group` construction needed for the low-tie row.

## Problems corrected in the focused artifact

1. **Hardcoded absolute MovieLens paths** were replaced by configuration-relative
   paths and an official-source downloader.
2. **No historical row retention** was addressed prospectively by saving frozen
   candidate rows and candidate scores/construction artifacts in every new run.
3. **No MovieLens row-level tie diagnostics** was addressed by applying the same
   diagnostic module to all regenerated scores.
4. **Weak permutation tests** were replaced by exact item-ranking equality checks for
   the hardened policy.
5. **Float32-key collision ambiguity** was isolated into an explicitly archived
   policy and paired with a uint64-plus-item-ID reference.
6. **Tiny floating-point jitter for random ties** was replaced by independent
   secondary keys sorted lexicographically under the unchanged primary score.
7. **Unfocused repository contents and identity leaks** were excluded.
8. **Paper/code naming ambiguity for the low-tie score** was documented and split
   into two explicit score names.
9. **Download and environment provenance** was added.
10. **Paper-output regeneration and tolerance checks** were added.

## Remaining distinction

The focused artifact is a prospective regeneration package, not a claim that missing
historical rows were recovered. Its archived directory preserves what was available;
its regeneration pipeline creates the complete evidence bundle that should be
retained for future review and publication.

## Additional audit findings surfaced in this package

11. **BPC `raw_count` was not a literal event count.** The focused artifact preserves
    the rating-weighted equation and calls out the manuscript wording mismatch.
12. **The retained MovieLens `experiment_summary.json` and paired tie-audit CSV came
    from nearby but non-identical executions.** Table 2 is tied explicitly to the
    paired stable/hash audit CSV.
13. **Paper-scale evaluator code created large whole-dataset ranking temporaries.**
    The focused artifact adds a tested bounded-memory evaluator and bounded-memory
    policy-difference diagnostics while preserving results.
14. **Run outputs lacked a complete cryptographic chain.** New runs now write the
    resolved configuration and SHA-256 integrity manifests, and can be replayed from
    frozen rows without resampling.
15. **Paper claims had no direct file index.** The artifact adds a machine-readable
    claim-to-evidence map and verifies that all major quantitative claims are mapped.
