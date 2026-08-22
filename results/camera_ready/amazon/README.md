# Amazon camera-ready summaries

These files summarize the evaluator-only analysis on 30,000 deterministically
reconstructed Amazon Beauty & Personal Care candidate rows, with 31 candidates per
row.

- `policy_comparison_for_paper.csv`: input-order, historical float32, hardened uint64,
  exact expected-over-ties, and 100-seed summaries for NDCG@10, Hit@10, and MRR.
- `metrics_by_policy.csv`: complete policy-level metric output.
- `randomized_summary.csv`: mean, standard deviation, standard error, minimum, and
  maximum over 100 independent hash seeds.
- `diagnostics.json`: exact tie counts, including separate counts for any tie, a tie
  containing the relevant item, and a tie crossing the top-10 boundary.
- `float32_collision_summary.json`: observed collision and permutation audit.
- `reproduction_check.csv`: 84 original-versus-reconstructed aggregate checks; all
  rows pass with absolute difference zero.

The frozen candidate rows, candidate-score matrices, and per-seed CSV files are in the
companion release asset documented in `results/camera_ready/README.md`.
