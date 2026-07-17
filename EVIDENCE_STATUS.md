# Evidence status

This repository is a transparent evaluation-audit package, not a claim of end-to-end historical reproduction. It separates three evidence classes that should not be merged: values printed in the manuscript, locally present derivative or current snapshots, and configuration facts reconstructed from the archived scripts and metadata.

## Reviewer claim-to-file map

| Claim or check | Evidence file | What the file establishes | Boundary |
|---|---|---|---|
| BPC RawCount NDCG@10 changes from 0.8474 to 0.1702, and Hit/Recall@10 from 0.9997 to 0.3526 | `results/manuscript_table2.csv` | Exact values displayed in manuscript Table 2 | Aggregate transcription only; no frozen rows or unrounded values |
| BPC Centered count and low-tie residualized controls behave as reported | `results/manuscript_table2.csv` | Exact displayed NDCG@10 stable/keyed values and changes | Does not verify internal score aliases or regenerate rows |
| BPC RawCount has a 100% any-tie rate, 99.94% boundary-tie rate at k=10, and 98.84% positive-in-tie rate; the residualized control has a 0.40% any-tie rate | `results/manuscript_bpc_tie_diagnostics.csv` | Exact aggregate rates stated in manuscript Section 5.1 | Row-level diagnostic arrays are absent |
| MovieLens shows a large RawCount and Centered-count stable/keyed change while popularity is nearly unchanged | `results/manuscript_table2.csv` | Exact values displayed in manuscript Table 2 | The historical paired result artifact is absent |
| A secondary BPC uncertainty summary is present locally | `results/local_bpc_tie_uncertainty_snapshot.csv` | The numeric contents of the locally present derivative CSV | Its referenced original tie-artifact output directory is absent; the legacy random-tie columns are not a valid uniform-tie estimator |
| The BPC manuscript and derivative summary use different delta conventions and retain different precision | `results/bpc_manuscript_vs_local.csv` | RawCount endpoints agree as displayed; the centered derivative stores 0.5477 while its displayed endpoints differ by 0.5476 | Does not prove source identity or recover unrounded manuscript values |
| A current MovieLens keyed-only result is present locally | `results/local_movielens_keyed_snapshot.csv` | Exact contents of the locally present `ranking_results.csv` | It is not the manuscript's historical paired stable/keyed artifact |
| The current MovieLens keyed values are close to, but not identical to, the manuscript keyed values | `results/movielens_manuscript_vs_local.csv` | Direct arithmetic differences between the two available sources | It does not identify why the sources differ |
| Seeds, row construction, k, data counts, labels, thresholds, splits, user selection, and negative exclusions are explicit where recoverable | `configs/bpc_audit.json`, `configs/movielens_audit.json` | A machine-readable reconstruction with literal `unknown` entries | These are evidence-oriented configs, not frozen run manifests |
| Provenance class and claim limits for every result table | `results/source_manifest.csv` | Which file is manuscript-reported, locally preserved, or derived | Does not elevate a derivative snapshot into primary evidence |

## Material evidence gaps

- **Frozen rows are absent.** The archive does not contain the historical per-row candidate identities, relevance labels, primary scores, or row permutations needed to replay the central same-row intervention directly.
- **The original `outputs/tie_artifact_experiment/` directory is absent.** In particular, the referenced `metrics_by_tie_mode.csv`, `per_model_metric_deltas.csv`, `tie_diagnostics.csv`, and random-tie run output are not present. The BPC claim is therefore supported here by exact manuscript aggregates plus a separate derivative summary, not by its original aggregate source files.
- **BPC is exact-aggregate evidence only.** The reported values and rates are preserved at the precision stated in the manuscript, but the missing original outputs and frozen rows prevent independent recomputation or recovery of additional precision.
- **The local BPC summary is a separate derivative source.** It defines delta as stable minus hash, whereas manuscript Table 2 defines change as keyed minus stable. For Centered count it stores a delta magnitude of 0.5477 even though the four-decimal endpoints differ by 0.5476. Both records are preserved as found; this package does not silently reconcile or overwrite either value.
- **The historical MovieLens paired stable/keyed artifact is absent.** The locally present `ranking_results.csv` is keyed only. Its keyed values differ slightly from the manuscript: RawCount NDCG@10 by +0.0004924146, Centered-count NDCG@10 by +0.0004377542, Popularity NDCG@10 by -0.0008556179, and RawCount Hit@10 by +0.0006 (local minus manuscript). These are distinct evidence sources, not rounding-equivalent copies.
- **MovieLens row diagnostics are absent.** No historical arrays or aggregates establish its any-tie, boundary-tie, positive-in-tie, top-k-set-change, or positive-rank-change rates. This package does not infer those rates from metric changes.
- **The archived float32-key collision audit is absent.** The historical keyed implementations normalize a mixed 64-bit value and cast it to float32. No preserved collision counts or collision-affected row trace are available, so the package does not claim collision-free historical execution.
- **The two retained domains do not use byte-identical sorting calls.** BPC uses a secondary-key `argsort` followed by a stable primary-score sort; the retained MovieLens script uses `lexsort`. They are equivalent when float32 secondary keys are unique, but their collision behavior is not proven identical. `archived_float32` preserves the manuscript-described BPC two-pass form.
- **The legacy BPC random-tie summary should not be cited.** The retained source script adds tiny jitter to float32 primary scores instead of sorting by an independent secondary key. At ordinary score magnitudes, `1e-10` perturbations can round away and fail to break exact ties. The CSV is preserved as provenance only; the executable artifact uses independent secondary keys and validates them against the analytic expectation.

## Safe interpretation

The package is sufficient for a reviewer to audit the stated formulas, evaluator logic, manuscript-level aggregates, explicit configuration choices, and the provenance mismatch between historical and currently present results. It is not sufficient to claim a byte-for-byte rerun of the historical experiments, to recover unreported precision, or to validate the central intervention at row level without restoring the frozen rows or regenerating them from the exact data snapshot and environment.
