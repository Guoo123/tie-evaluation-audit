# Regenerated run output schema

Each completed dataset run writes to `results/regenerated/<dataset>/`.

| File | Role |
|---|---|
| `resolved_config.json` | Exact effective configuration with loader-only absolute paths removed |
| `runtime_manifest.json` | Python, package, platform, CPU, config hash, and checkout commit information |
| `frozen_candidate_rows.npz` | `user_indices`, `positive_items`, and the positive-first candidate matrix |
| `candidate_scores_<score>.npy` | Primary scores aligned one-to-one with the frozen candidate matrix |
| `metrics_by_policy.csv` | Stable, archived-keyed, hardened-keyed, and analytic metrics |
| `randomized_<score>.csv` | Per-seed randomized within-tie metrics |
| `diagnostics.json` | Exact-tie rates and stable-versus-hardened row-level sensitivity diagnostics |
| `run_summary.json` | Dataset counts, run scale, score families, runtime, and output index |
| `run_integrity_manifest.json` | SHA-256 and size for every retained run output plus upstream manifests |
| `replay/metrics_by_policy.csv` | Independently recomputed metrics from frozen rows and scores |
| `replay/replay_manifest.json` | Hashes of replay inputs and comparison against the original run metrics |

## Frozen-row contract

For every row `r`:

1. `candidates[r, 0] == positive_items[r]`;
2. the positive identity occurs exactly once;
3. all policies consume the same candidate and score arrays;
4. only secondary ordering inside exact primary-score ties may differ; and
5. `hardened_uint64` must return exactly the same ranked item sequence after any
   permutation of the row representation.

`python scripts/verify_artifact.py --level full` enforces this contract and compares
full-scale regenerated values with the archived targets.
