# Expected results

The canonical targets are machine-readable in `configs/paper.yaml` and
`results/archived/paper_table2.csv`.

## Submitted aggregate table

| Dataset | Score | Stable NDCG@10 | Archived keyed NDCG@10 |
|---|---|---:|---:|
| BPC | RawCount | 0.8474 | 0.1702 |
| BPC | centered count | 0.7103 | 0.1627 |
| BPC | residualized group control | 0.1689 | 0.1685 |
| MovieLens | RawCount | 0.8380 | 0.2332 |
| MovieLens | centered count | 0.8406 | 0.2341 |
| MovieLens | popularity | 0.5905 | 0.5905 |

RawCount Hit@10 changes from 0.9997 to 0.3526 on BPC and from 0.9390 to
0.4358 on MovieLens.

## BPC tie diagnostics retained from the submitted audit

- RawCount any-tie rate: 1.0000;
- RawCount boundary-tie rate at 10: 0.9994;
- RawCount positive-in-tie rate: 0.9884;
- centered-count any-tie rate: 1.0000;
- low-tie control any-tie rate reported in the writer pack: 0.0040.

## Interpreting a regenerated run

A successful run should satisfy all of the following:

1. processed dataset counts equal the configured targets;
2. stable tie-heavy metrics are much larger than archived-keyed, hardened, and
   analytic metrics;
3. low-tie controls and popularity change minimally;
4. hardened rankings survive arbitrary row permutations exactly;
5. analytic and repeated-randomized estimates agree within Monte Carlo error; and
6. the submitted stable/archived-keyed values fall within configured tolerances.

A regeneration can still validate the mechanism when aggregate values move slightly,
but any movement must be investigated through the saved data manifest, processed
counts, candidate rows, score hashes, and package versions rather than silently
accepted.
