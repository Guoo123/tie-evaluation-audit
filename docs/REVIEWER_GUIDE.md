# Reviewer guide

## 1. Verify the package before execution

```bash
python scripts/verify_manifest.py
python scripts/check_package_hygiene.py
python scripts/preflight.py
```

The first command verifies every distributable file against
`ARTIFACT_MANIFEST.sha256`. The second rejects operating-system metadata (including
AppleDouble `._*` files), symbolic links, unexpected generated/downloaded files,
unmanifested distributable files, user-specific home paths, and stale anonymous-review
wording while confirming that the public citation metadata is present. Author names,
emails, and the hosted repository URL are intentionally retained for FRAME Track 2
single-blind review. Preflight reports Python/package compatibility, free disk, and
whether each public dataset is already present.

## 2. Verify the mechanism without external data

```bash
pytest
python scripts/verify_artifact.py --level smoke
```

The checks require all of the following:

- one positive followed by 30 all-tied negatives gives stable Hit@10 and NDCG@10 of 1;
- uniform ordering inside that tie gives expected Hit@10 of `10/31` and NDCG@10 of about `0.1466`;
- analytic formulas match enumeration;
- stable, archived-keyed, and hardened policies agree when primary scores are strict;
- hardened rankings are exactly equal after independent row permutations;
- bounded-memory and full-matrix evaluation agree; and
- toy BPC and MovieLens pipelines save rows, scores, metrics, configuration, hashes,
  and replayable output.

## 3. Audit the submitted numerical claims

```bash
python scripts/verify_artifact.py --level archive
```

The verifier prints JSON to the terminal without modifying the checkout. Add
`--output PATH` only when a saved verification report is wanted.

Start with:

```text
results/archived/CLAIM_EVIDENCE.json
results/archived/paper_table2.csv
results/archived/PROVENANCE.json
docs/CLAIM_EVIDENCE_MATRIX.md
docs/ARCHIVED_EVIDENCE_AUDIT.md
```

The verifier checks that the eight paper-table rows agree with the retained BPC and
MovieLens aggregate files, that the BPC dataset counts match the declared targets,
and that the evidence map includes every major quantitative claim. It also verifies
the entire package manifest.

The evidence boundary is explicit: the supplied development ZIP did not contain the
old frozen candidate rows. Historical aggregate checking is therefore not mislabeled
as row-level replay.

## 4. Regenerate MovieLens from public data

```bash
python scripts/run_movielens.py --download --config configs/paper.yaml
python scripts/replay_frozen.py --dataset movielens --config configs/paper.yaml
python scripts/make_paper_outputs.py --config configs/paper.yaml
python scripts/verify_artifact.py --level full --config configs/paper.yaml
```

The output should include:

```text
results/regenerated/movielens/
  resolved_config.json
  runtime_manifest.json
  frozen_candidate_rows.npz
  candidate_scores_raw_count.npy
  candidate_scores_centered_count.npy
  candidate_scores_popularity.npy
  metrics_by_policy.csv
  diagnostics.json
  randomized_*.csv
  run_summary.json
  run_integrity_manifest.json
  replay/replay_manifest.json
```

`verify_artifact.py --level full` requires the exact declared dataset counts, checks
the configured numerical tolerance against the submitted stable/archived-keyed
values, tests row permutation on saved score matrices, replays all deterministic and
analytic metrics, and validates the run-integrity manifest.

## 5. Audit or regenerate BPC

The code path is deliberately staged:

1. `src/tie_eval/download.py` downloads the two official compressed category files.
2. `src/tie_eval/bpc_preprocess.py` streams JSONL to Parquet, resolves duplicates,
   joins metadata, performs sequential iterative 3-core filtering, maps sorted IDs,
   and creates per-user train/validation/test splits.
3. `src/tie_eval/labels.py` constructs the three binary product attributes.
4. `src/tie_eval/bpc_models.py` builds rating-weighted raw/centered scores, masked-text
   propensity residuals, and the cross-fitted user/group low-tie control.
5. `src/tie_eval/sampling.py` samples one positive plus 30 history-safe negatives.
6. `src/tie_eval/bpc.py` freezes rows/scores and runs every tie policy and diagnostic.

Paper-scale staged execution:

```bash
python scripts/download_data.py --dataset bpc
python scripts/run_bpc.py --stage stage_raw
python scripts/run_bpc.py --stage preprocess
python scripts/run_bpc.py --stage model
python scripts/run_bpc.py --stage evaluate
python scripts/replay_frozen.py --dataset bpc
python scripts/verify_artifact.py --level full
```

A resource-reduced evaluator run can use:

```bash
python scripts/run_bpc.py --stage evaluate --max-rows 10000 --skip-low-tie-control
```

The output is explicitly marked as a subset and is not compared with paper-scale
metrics.

## Direct answers to likely reviewer questions

### Were candidates or primary scores changed between Stable and Keyed?

No. Each new run writes one frozen candidate matrix and one aligned score matrix per
score family. Every tie policy consumes exactly those arrays.

### Is the submitted Keyed column strictly row-order invariant?

Not unconditionally. It used the archived float32 secondary key; rare float32 key
collisions can fall back to stable order. The artifact preserves that policy for
number compatibility and separately tests `hardened_uint64`, which adds item identity
as a collision fallback and is strictly invariant.

### Does one keyed result estimate the average over tie orders?

No. It is one reproducible realization. The artifact additionally reports the analytic
single-positive expectation and repeated randomized estimates.

### Which BPC score produced the low-tie row?

`residualized_group_control`, corresponding to the development name `po_group`. It is
not the simpler item-residual-only `sre` score. The exact formula and archived values
for both are documented.

### Is BPC RawCount literally an unweighted count?

No. The operational code weights each history event by the scaled rating. The artifact
preserves the historical code name but gives the exact equation.

### Why do two retained MovieLens files have slightly different hash-mode values?

They came from nearby but non-identical executions. The paper table uses the dedicated
paired stable/hash tie-audit CSV. The experiment summary is retained for dataset counts
and context. No discrepancy is silently erased.
