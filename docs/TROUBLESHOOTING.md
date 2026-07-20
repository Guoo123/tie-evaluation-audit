# Troubleshooting

## MovieLens checksum failure

Delete `data/raw/movielens/ml-25m.zip` and rerun with `--download`. Do not disable the
MD5 check. A changed upstream archive should be investigated rather than silently
accepted.

## BPC count mismatch

Inspect:

- `data/raw/bpc/download_manifest.json`;
- `data/interim/bpc/stage_manifest.json`;
- `data/processed/bpc/dataset_summary.json`; and
- the `kcore_history` array in that summary.

A count mismatch usually indicates an upstream file change, an interrupted staged
Parquet file, or a difference in duplicate canonicalization. Use `--force` to rebuild
stages only after preserving the mismatching manifests.

## Out-of-memory during BPC group control

Run the main tie-heavy scores first:

```bash
python scripts/run_bpc.py --stage model --skip-low-tie-control
python scripts/run_bpc.py --stage evaluate --skip-low-tie-control
```

The RawCount and centered-count claims do not depend on the expensive group control.
Run that control on a larger-memory machine, reusing all prior stages.

## Metrics differ but dataset counts match

Compare the regenerated `runtime_manifest.json`, candidate-row hash, and score
artifacts. MovieLens differences can arise from Python set iteration or BLAS versions;
the pinned Python 3.11 environment minimizes this. BPC text-model differences can
arise from scikit-learn or sparse linear-algebra versions. The configured tolerance is
applied only after count checks.

## Verification says “subset-not-compared-to-paper”

This is expected after `--max-rows`. Remove that option for a full-paper comparison.
