# Camera-ready evidence

This directory contains compact, text-based evidence added for the FRAME 2026
camera-ready version. It does not replace `results/archived/`, which preserves the
aggregate evidence used in the accepted submission.

The Amazon camera-ready analysis distinguishes three evidence layers:

1. **Accepted aggregate evidence**: the original stable and historical float32-key
   values under `results/archived/`.
2. **Deterministic reconstruction**: 30,000 Amazon rows and six candidate-score
   matrices reconstructed from preserved historical inputs. All 84 checked aggregate
   values match the original run exactly.
3. **New evaluator-only analysis**: hardened uint64 tie-breaking, the exact metric
   expectation under uniform random tie-breaking, 100 independent hash seeds, a
   float32-collision audit, and permutation-invariance checks.

The summary files required to check the camera-ready tables are committed under
`amazon/`. The row-level arrays and per-seed files are intentionally distributed as a
companion GitHub Release asset because they are binary and substantially larger than
the source repository.

Expected release asset:

```text
FRAME_2026_camera_ready_artifact_release_payload_updated.zip
SHA-256: 4b1e0b69e5d2adad9a0601ce28d79e5ae7c393e600f5ae0f7ebf4b924716ee18
```

For MovieLens, the camera-ready paper reports the aggregate values unchanged from the
accepted version. The canonical paired aggregate source is retained at
`results/archived/movielens/tie_audit_summary.csv`, but the corresponding row-level
candidate and score arrays were not retained; no camera-ready row-level MovieLens
replay is claimed.
