# Claim-to-evidence matrix

This document tells a reviewer exactly where each quantitative manuscript claim is
supported and what a fresh run will produce. The machine-readable version is
`results/archived/CLAIM_EVIDENCE.json`.

| Manuscript claim | Historical evidence retained in this package | Fresh-run evidence |
|---|---|---|
| BPC has 2,073,945 users, 389,478 items, 7,528,500 train rows, and one validation/test row per user | `results/archived/bpc/dataset_summary.json` | `results/regenerated/bpc/run_summary.json` and `data/processed/bpc/dataset_summary.json` |
| BPC RawCount NDCG@10 is 0.8474 under stable positive-first order and 0.1702 under the archived key | `results/archived/bpc/tie_audit_summary.csv`; normalized into `paper_table2.csv` | `results/regenerated/bpc/metrics_by_policy.csv` |
| BPC RawCount Hit@10 is 0.9997 versus 0.3526 | same BPC tie-audit CSV | same regenerated metrics CSV |
| BPC RawCount has any-tie rate 1.0000, boundary-tie rate 0.9994, and positive-in-tie rate 0.9884 | same BPC tie-audit CSV | `results/regenerated/bpc/diagnostics.json` |
| BPC centered NDCG@10 is 0.7103 versus 0.1627 | same BPC tie-audit CSV | regenerated metrics CSV |
| BPC low-tie NDCG@10 is 0.1689 versus 0.1685, with any-tie rate 0.0040 | BPC row named `residualized_group_control` in the tie-audit CSV | regenerated score `residualized_group_control` |
| MovieLens uses 162,341 users, 13,816 Tag-Genome-covered items, 9,887,006 train rows, and 10,000 audited users | `results/archived/movielens/experiment_summary.json` | `results/regenerated/movielens/run_summary.json` |
| MovieLens RawCount NDCG@10 is 0.8380 versus 0.2332; Hit@10 is 0.9390 versus 0.4358 | `results/archived/movielens/tie_audit_summary.csv` and `ranking_results_by_tie_mode.csv` | regenerated metrics CSV |
| MovieLens centered NDCG@10 is 0.8406 versus 0.2341 | same MovieLens tie-audit CSV | regenerated metrics CSV |
| MovieLens popularity is unchanged to four decimals | same MovieLens tie-audit CSV | regenerated metrics CSV |
| One positive and 30 negatives, all tied, imply expected Hit@10 of 10/31 and expected NDCG@10 of about 0.1466 | analytic formula in the manuscript | `src/tie_eval/metrics.py`, `examples/toy/run_toy.py`, and unit tests |

## Evidence classes

The historical files support **aggregate claim verification**. They do not support
historical row-by-row replay because the supplied ZIP did not contain the old frozen
rows. A fresh run supports the stronger chain:

```text
public URL + checksum manifest
  -> processed-count manifest
  -> resolved configuration and environment
  -> frozen candidate rows
  -> candidate score matrices
  -> policy-specific rankings and diagnostics
  -> Table 2 / Figure 1 data
```

The verifier never labels aggregate evidence as row-level replay evidence.
