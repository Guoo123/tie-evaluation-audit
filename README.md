# Anonymous artifact: order-invariance audit for tie-heavy recommendation scores

This is the focused, anonymous artifact for **“Tie Handling Is Part of the Evaluation
Protocol: An Order-Invariance Audit for Tie-Heavy Recommender Scores.”** It was
reconstructed from the supplied development ZIP so that a reviewer can inspect the
retained paper evidence and independently regenerate a new run from public data.

The repository supports four distinct operations:

| Operation | External data? | What it establishes |
|---|---:|---|
| `pytest` + smoke verification | No | formulas, failure mechanism, sampler invariants, and hardened row-permutation invariance |
| archived verification | No | paper-table arithmetic, retained dataset counts, claim-to-file provenance, and package integrity |
| end-to-end regeneration | Yes | fresh download, preprocessing, split, scoring, resampling, tie audit, and paper outputs |
| frozen-row replay | No, after a run | exact evaluator replay from saved candidates and candidate scores |

“Regeneration” is not represented as historical row-level replay. The supplied ZIP
did not contain the old frozen candidate rows for either dataset. The files under
`results/archived/` therefore support aggregate claim verification. A new run creates
and hashes the missing row-level objects so it can be replayed exactly thereafter.
See `docs/ARCHIVED_EVIDENCE_AUDIT.md` and
`results/archived/CLAIM_EVIDENCE.json`.

## Reviewer path without downloading data

Use Python 3.11 and the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

python scripts/preflight.py
python scripts/verify_manifest.py
pytest
python scripts/verify_artifact.py --level archive
python scripts/check_anonymity.py
```

This checks the all-tied example, analytic expectations, no-tie equivalence,
hardened permutation invariance, bounded-memory and full-matrix evaluator
agreement, toy end-to-end pipelines, archived evidence, and all distributable file
hashes.

## Recommended full reviewer run: MovieLens

MovieLens 25M is the practical end-to-end reviewer target because the official archive
is fixed, moderate in size, and includes ratings and Tag Genome files.

```bash
python scripts/preflight.py --dataset movielens
python scripts/run_movielens.py --download --config configs/paper.yaml
python scripts/replay_frozen.py --dataset movielens --config configs/paper.yaml
python scripts/make_paper_outputs.py --config configs/paper.yaml
python scripts/verify_artifact.py --level full --config configs/paper.yaml
```

The pipeline:

1. downloads the official `ml-25m.zip` and adjacent checksum file, then verifies the published MD5;
2. selects the six declared Tag Genome concepts at threshold 0.7;
3. retains ratings at least 4.0 for Tag-Genome-covered movies;
4. recreates the archived global chronological 80/10/10 split;
5. selects the first 10,000 users in the test segment and one positive per user;
6. samples 30 unique training-history-safe negatives with seed 42;
7. freezes the positive-first candidate rows and every candidate-score matrix;
8. evaluates stable, archived-keyed, hardened-keyed, analytic, and repeated-randomized tie treatments; and
9. writes row-level diagnostics, environment/configuration records, and SHA-256 manifests.

## Amazon Beauty & Personal Care regeneration

The BPC run is substantially larger and is staged and resumable:

```bash
python scripts/preflight.py --dataset bpc
python scripts/download_data.py --dataset bpc
python scripts/run_bpc.py --stage stage_raw
python scripts/run_bpc.py --stage preprocess
python scripts/run_bpc.py --stage model
python scripts/run_bpc.py --stage evaluate
python scripts/replay_frozen.py --dataset bpc
python scripts/make_paper_outputs.py
python scripts/verify_artifact.py --level full
```

The equivalent one-command data-to-results run is:

```bash
python scripts/run_bpc.py --download --stage all
```

A reviewer with limited resources can omit the expensive low-tie group control:

```bash
python scripts/run_bpc.py --stage all --skip-low-tie-control
```

Or, after preprocessing/model construction, test only the evaluator on a prefix:

```bash
python scripts/run_bpc.py --stage evaluate --max-rows 10000 --skip-low-tie-control
```

Subset metrics are labeled as such and are never compared with the paper-scale table.

## Exact score definitions

### BPC historical `raw_count`

The development identifier is retained for compatibility, but the operational
quantity is rating weighted:

```text
y[r] = clip((rating[r] - 1) / 4, 0, 1)
S_raw[u,l] = sum_{training rows r of user u} y[r] * A[item[r],l]
score_raw(u,i) = S_raw[u] dot A[i]
```

It should be described as a **rating-weighted historical attribute total**, not as a
literal unweighted event count.

### BPC low-tie control

The paper row `0.1689 -> 0.1685` corresponds to the development score `po_group`,
renamed here as `residualized_group_control`. It combines item-side attribute
residuals from three-fold masked-text propensity models with a cross-fitted
user-plus-item-group outcome control. The simpler item-residual-only score is retained
separately as `item_residual_only`; it is not the source of the reported low-tie row.

### MovieLens RawCount

MovieLens uses unweighted training-history counts of six thresholded Tag Genome
attributes, followed by a user-count/item-attribute dot product.

Full equations and implementation locations are in `docs/PAPER_TO_CODE.md`.

## Tie policies

| Policy | Meaning | Row-order invariant? | Submitted table? |
|---|---|---:|---:|
| `stable_positive_first` | stable descending primary-score sort; input order resolves ties | No | Stable column |
| `archived_float32` | historical 64-bit mixer converted to float32 before sorting | Not guaranteed under rare float32 collisions | Keyed column |
| `hardened_uint64` | `(-score, uint64_key, item_id)` | Yes | Added reference |
| `analytic_uniform_tie_expectation` | exact expected metric over positive positions in its tie block | Yes, as an expectation | Added reference |
| `randomized` | repeated independent continuous secondary keys | In expectation | Added sensitivity analysis |

The archived policy is retained only for submitted-number compatibility. The hardened
policy is the strict implementation of row-order invariance.

## Output and replay contract

A completed dataset run writes:

```text
results/regenerated/<dataset>/
  resolved_config.json
  runtime_manifest.json
  frozen_candidate_rows.npz
  candidate_scores_<score>.npy
  metrics_by_policy.csv
  diagnostics.json
  randomized_<score>.csv
  run_summary.json
  run_integrity_manifest.json
  replay/
    metrics_by_policy.csv
    diagnostics.json
    replay_manifest.json
```

`run_integrity_manifest.json` hashes the retained run outputs and upstream manifests.
`replay_manifest.json` hashes the frozen replay inputs and requires replayed metrics
to match the recorded run metrics. See `docs/RUN_OUTPUT_SCHEMA.md`.

## Repository map

```text
configs/paper.yaml                    paper parameters, seeds, targets, tolerances
src/tie_eval/                         data, scoring, ranking, metrics, diagnostics
scripts/download_data.py              official-source downloads and checksums
scripts/run_movielens.py               full MovieLens regeneration
scripts/run_bpc.py                     staged BPC regeneration
scripts/replay_frozen.py               evaluator-only replay from saved rows/scores
scripts/make_paper_outputs.py          Table 2 and Figure 1 data products
scripts/verify_artifact.py             smoke, archive, and full verification
scripts/preflight.py                   environment, disk, RAM, and data readiness
results/archived/                      retained aggregate evidence and provenance
results/regenerated/                   locally generated outputs; not distributed
examples/toy/                          data-free mechanism example
tests/                                 unit and toy end-to-end tests
docs/                                  reviewer, data, evidence, and limitation notes
```

## Resource guidance

| Run | RAM | Free disk | Typical CPU time |
|---|---:|---:|---:|
| tests and archived verification | <2 GB | <1 GB | under a few minutes |
| MovieLens full regeneration | 12–24 GB | 5–10 GB | roughly 10–40 minutes |
| BPC raw/centered paper-scale audit | 32–64 GB | 40–80 GB | several hours |
| BPC with cross-fitted group control | 64–96 GB recommended | 60–100 GB | several hours to more than one day |

Runtime depends on storage and CPU throughput. BPC downloads, staged Parquet files,
processed splits, propensity outputs, control predictions, and score artifacts are
reused unless `--force` is supplied.

## Validation boundary of this distributed ZIP

The artifact-construction environment validated the code with unit tests and toy
end-to-end runs, verified the retained aggregate evidence, rebuilt the paper table and
figure from archived evidence, checked anonymity, and verified the package checksum
manifest. It did not complete the public-data MovieLens or BPC numerical regeneration
because external dataset downloads were unavailable in that environment. The package
therefore does not claim an unobserved full-data run; the exact commands and expected
checks are provided for reviewer execution.

Start with `docs/REVIEWER_GUIDE.md`. The most important evidence caveats are in
`docs/KNOWN_LIMITATIONS.md`.
