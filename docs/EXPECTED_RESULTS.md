# Expected results

The accepted aggregate targets remain machine-readable in `configs/paper.yaml` and
`results/archived/paper_table2.csv`. The camera-ready evaluator-only summaries are in
`results/camera_ready/amazon/` and use `configs/camera_ready.yaml`.

## Accepted cross-domain aggregate comparison

| Dataset | Score | Input-order NDCG@10 | Hash tie-break NDCG@10 |
|---|---|---:|---:|
| Amazon Beauty | rating-weighted attribute overlap | 0.8474 | 0.1702 |
| Amazon Beauty | centered attribute overlap | 0.7103 | 0.1627 |
| Amazon Beauty | residualized attribute score | 0.1689 | 0.1685 |
| MovieLens | tag-attribute overlap | 0.8380 | 0.2332 |
| MovieLens | centered tag overlap | 0.8406 | 0.2341 |
| MovieLens | item popularity | 0.5905 | 0.5905 |

For the primary overlap scores, Hit@10 changes from 0.9997 to 0.3526 on Amazon and
from 0.9390 to 0.4358 on MovieLens.

## Camera-ready Amazon policy comparison

| Score | Input order | Hardened hash | Expected over ties | 100 hashes, mean ± SD |
|---|---:|---:|---:|---:|
| rating-weighted attribute overlap | 0.8474 | 0.1702 | 0.1693 | 0.1694 ± 0.0012 |
| centered attribute overlap | 0.7103 | 0.1627 | 0.1614 | 0.1615 ± 0.0011 |
| residualized attribute score | 0.1689 | 0.1685 | 0.1685 | 0.16851 ± 0.00003 |

Full-precision values are in
`results/camera_ready/amazon/policy_comparison_for_paper.csv`.

## Amazon tie diagnostics

For rating-weighted attribute overlap on 30,000 rows:

- rows with any exact tie: 30,000 (100.00%);
- rows with a tie crossing the top-10 boundary: 29,981 (99.94%);
- rows with the relevant item in a tie: 29,651 (98.84%).

For the residualized diagnostic score:

- rows with any exact tie: 117 (0.39%);
- rows with a tie crossing the top-10 boundary: 13 (0.043%);
- rows with the relevant item in a tie: 19 (0.063%).

The exact counts are in `results/camera_ready/amazon/diagnostics.json`.

## Reconstruction and collision checks

- all 84 original-versus-reconstructed Amazon aggregate checks pass;
- maximum absolute reconstruction difference: 0.0;
- one row contains a distinct-item float32 secondary-key collision inside a relevant
  primary-score tie for the overlap scores;
- the collision changes neither top-10 membership nor the relevant item’s rank;
- the hardened uint64 policy has zero permutation-invariance failures.

## Interpreting a new public-data run

A successful full regeneration should satisfy all of the following:

1. processed dataset counts equal the configured targets;
2. input-order metrics for tie-heavy scores are much larger than hash, hardened, and
   analytic metrics;
3. low-tie diagnostic scores and item popularity change minimally;
4. hardened rankings survive arbitrary row permutations exactly;
5. analytic and repeated-randomized estimates agree within Monte Carlo error; and
6. accepted input-order/historical-key values fall within configured tolerances.

A prospective regeneration can validate the mechanism even when upstream data or
package changes move aggregate values slightly. Such movement must be investigated
through manifests, processed counts, candidate-row hashes, score hashes, and package
versions rather than silently accepted.
