# Camera-ready provenance

This note separates the evidence used in the accepted FRAME 2026 submission from the
additional evaluator-only evidence reported in the camera-ready paper.

## 1. Accepted aggregate evidence

`results/archived/` contains the aggregate files retained from the accepted
submission. The stable/input-order and historical float32-key values in the
cross-domain table are unchanged.

For MovieLens, the camera-ready manuscript reports those accepted aggregate values
unchanged. A targeted search of the original VM did not recover the canonical paired
source file or row-level candidate and score arrays. A nearby retained Tag Genome run
contains different values and is not substituted for the accepted source. The
repository therefore makes no claim of exact row-level replay for the accepted
MovieLens comparison.

## 2. Deterministically reconstructed Amazon evidence

Preserved historical Amazon inputs and the original deterministic sampling settings
were sufficient to reconstruct:

- 30,000 evaluation rows;
- 31 candidates per row;
- one relevant item in the first input position;
- candidate sampling seed 42; and
- six aligned candidate-score matrices.

The reconstruction was checked against the original run at the aggregate level. All
84 comparisons in `results/camera_ready/amazon/reproduction_check.csv` pass with
maximum absolute difference `0.0`.

The reconstructed rows are not described as files that were frozen during the
original run. They are labeled as deterministic reconstructions from preserved
historical inputs.

## 3. New camera-ready evaluator-only analysis

No model was retrained, no candidate was resampled, and no primary score was changed.
On the reconstructed Amazon rows, the camera-ready analysis adds:

- hardened unsigned-64-bit deterministic hash tie-breaking;
- exact expected Hit Rate, NDCG, and MRR under uniform random tie-breaking;
- 100 independent hash seeds;
- a float32 secondary-key collision audit; and
- empirical permutation-invariance checks.

The summary outputs are committed under `results/camera_ready/amazon/`.
The binary candidate rows, score matrices, per-seed CSV files, replay scripts, and
full provenance package are distributed as the companion release asset documented in
`results/camera_ready/README.md`.

## 4. Key verified values

For the Amazon rating-weighted attribute-overlap score:

| Treatment | NDCG@10 |
|---|---:|
| Input-order tie-breaking | 0.847419307899178 |
| Historical float32 key | 0.17023696678473144 |
| Hardened uint64 key | 0.17023696678473144 |
| Exact expectation over tie orders | 0.16930035129969778 |
| 100-seed mean | 0.16942719461148026 |
| 100-seed standard deviation | 0.00116031048339268 |

Tie prevalence on the same score arrays is:

- rating-weighted overlap: 30,000/30,000 rows with any exact tie, 29,981/30,000 with a
  top-10 boundary tie, and 29,651/30,000 with the relevant item in a tie;
- residualized diagnostic score: 117/30,000 rows with any exact tie, or 0.39%.

The observed float32 collision affected one row for the tie-heavy overlap scores. It
did not change top-10 membership, the relevant item rank, or any aggregate metric. The
hardened uint64 policy had zero row-order-dependence failures in the permutation test.

## 5. Configuration boundary

- `configs/paper.yaml` preserves the accepted-submission settings, including the
  original 20-run randomized check and the historical rounded 0.40% diagnostic target.
- `configs/camera_ready.yaml` records the final 100-seed analysis and the verified
  117/30,000 = 0.39% residualized-score tie rate.

This separation prevents the camera-ready additions from being presented as if they
were part of the original submission run.
