# Artifact: order-invariance audit for tie-heavy recommendation scores

This repository is the reproducibility artifact for **“Tie Handling Is Part of the
Evaluation Protocol: An Order-Invariance Audit for Tie-Heavy Recommender Scores.”**
It supports mechanism tests, aggregate-evidence verification, public-data
regeneration, and evaluator-only replay from frozen candidate rows and scores.

## Camera-ready update

The accepted paper compared input-order tie-breaking with a deterministic secondary
key. The camera-ready version keeps those aggregate results unchanged and adds three
Amazon evaluator-only analyses on the same 30,000 rows and candidate scores:

- a hardened unsigned-64-bit hash tie-break with item ID as the final fallback;
- the exact metric expectation under uniform random tie-breaking; and
- 100 independent hash seeds, reported with mean and variation.

Preserved historical inputs deterministically reconstruct 30,000 Amazon Beauty &
Personal Care rows with 31 candidates per row and six score matrices. The
reconstruction reproduces all 84 checked aggregate values from the original run
exactly. Compact summaries are committed under `results/camera_ready/amazon/`.
The frozen arrays and per-seed outputs are distributed as the companion GitHub Release
asset described in `results/camera_ready/README.md`.

For MovieLens, the camera-ready paper reports the aggregate values unchanged from the
accepted version. The canonical paired aggregate source is retained at
`results/archived/movielens/tie_audit_summary.csv`, but the corresponding row-level
candidate and score arrays were not retained, so no camera-ready row-level MovieLens
replay is claimed. See `docs/CAMERA_READY_PROVENANCE.md`.

## What the repository can verify

| Operation | External data? | What it establishes |
|---|---:|---|
| `pytest` + smoke verification | No | formulas, failure mechanism, sampler invariants, and hardened row-permutation invariance |
| archived verification | No | accepted-table arithmetic, retained dataset counts, claim-to-file provenance, and package integrity |
| camera-ready summary verification | No | the Amazon four-policy table, 100-seed summaries, tie diagnostics, collision audit, and exact 84-value reconstruction check |
| end-to-end regeneration | Yes | fresh download, preprocessing, split, scoring, resampling, tie audit, and paper outputs |
| frozen-row replay | No, after a run or with the release asset | evaluator replay from fixed candidates and candidate scores |

“Regeneration” is not represented as historical row-level replay. The original
submission ZIP did not contain frozen candidate rows for either dataset. The Amazon
camera-ready rows were later reconstructed deterministically from preserved historical
inputs and verified against the original aggregate outputs. The MovieLens evidence
remains aggregate-only.

## Fast verification without downloading data

For Python 3.11–3.13:

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux
# PowerShell: .\.venv\Scripts\Activate.ps1
# cmd.exe: .venv\Scripts\activate.bat
python -m pip install -U pip
python -m pip install -e ".[dev]"

python scripts/preflight.py
python scripts/verify_manifest.py
pytest
python scripts/verify_artifact.py --level archive
python scripts/check_package_hygiene.py
python scripts/make_paper_outputs.py --config configs/camera_ready.yaml
```

The paper-output command writes:

```text
results/regenerated/paper_outputs/
  table2_cross_domain.csv
  table3_amazon_policies.csv
  manifest.json
```

It no longer generates the redundant figure removed from the camera-ready paper.

## Exact frozen environment

For paper-scale numerical regeneration, use Python 3.11 with the exact versions in
`requirements.txt`, `environment.yml`, or the Dockerfile:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Use `configs/paper.yaml` to preserve the accepted-submission settings, including the
original 20-run randomized check. Use `configs/camera_ready.yaml` for the final
100-seed analysis and the corrected 0.39% Amazon residualized-score tie rate.

## Public-data regeneration

### MovieLens 25M + Tag Genome

```bash
python scripts/preflight.py --dataset movielens
python scripts/run_movielens.py --download --config configs/paper.yaml
python scripts/replay_frozen.py --dataset movielens --config configs/paper.yaml
python scripts/make_paper_outputs.py --config configs/camera_ready.yaml
python scripts/verify_artifact.py --level full --config configs/paper.yaml
```

The pipeline downloads the fixed MovieLens 25M archive, selects the six declared Tag
Genome concepts, recreates the archived global chronological split, samples 30 unique
training-history-safe negatives per selected user, freezes the candidate rows and
score matrices, and evaluates all tie policies.

A new MovieLens run is prospective regeneration. It does not replace the accepted
aggregate values or recover the missing historical row-level candidate and score
arrays.

### Amazon Beauty & Personal Care

The internal configuration key and command-line dataset name remain `bpc` for
backward compatibility:

```bash
python scripts/preflight.py --dataset bpc
python scripts/download_data.py --dataset bpc
python scripts/run_bpc.py --stage stage_raw
python scripts/run_bpc.py --stage preprocess
python scripts/run_bpc.py --stage model
python scripts/run_bpc.py --stage evaluate
python scripts/replay_frozen.py --dataset bpc
python scripts/make_paper_outputs.py --config configs/camera_ready.yaml
python scripts/verify_artifact.py --level full
```

The equivalent one-command data-to-results run is:

```bash
python scripts/run_bpc.py --download --stage all
```

A limited-resource run can omit the expensive low-tie diagnostic score:

```bash
python scripts/run_bpc.py --stage all --skip-low-tie-control
```

Subset outputs are labeled as such and are not compared with the paper-scale table.

## Score definitions

### Amazon rating-weighted attribute overlap

The historical implementation name is `raw_count`, but the operational quantity is
rating weighted:

```text
y[r] = clip((rating[r] - 1) / 4, 0, 1)
S_raw[u,l] = sum_{training rows r of user u} y[r] * A[item[r],l]
score_raw(u,i) = S_raw[u] dot A[i]
```

The paper therefore calls it **rating-weighted attribute overlap** rather than a
literal interaction count.

### Amazon residualized diagnostic score

The paper row `0.1689 -> 0.1685` corresponds to the development score `po_group`,
exposed here as `residualized_group_control`. It combines item-side attribute
residuals from three-fold masked-text propensity models with a cross-fitted user and
user-by-item-group outcome control. The simpler item-residual-only score remains
available as `item_residual_only`; it is not the source of the reported low-tie row.

### MovieLens tag-attribute overlap

MovieLens uses unweighted training-history counts of six thresholded Tag Genome
attributes, followed by a user-count/item-attribute dot product.

Full equations and implementation locations are in `docs/PAPER_TO_CODE.md`.

## Tie treatments

| Policy | Meaning | Row-order invariant? | Role |
|---|---|---:|---|
| `stable_positive_first` | stable descending score sort; input order resolves ties | No | accepted input-order column |
| `archived_float32` | historical 64-bit mixer converted to float32 before sorting | Not guaranteed under rare float32 collisions | accepted compatibility result |
| `hardened_uint64` | sort by `(-score, uint64_key, item_id)` | Yes | reference deterministic implementation |
| `analytic_uniform_tie_expectation` | exact expected metric over relevant-item positions in its tie block | Yes, as an expectation | camera-ready analysis |
| `randomized` | repeated independent secondary-key seeds | In expectation | camera-ready sensitivity analysis |

The finite Amazon collision audit found one row with a distinct-item float32 collision
inside a relevant score tie. It changed the internal ordering of tied items but did
not change top-10 membership, the relevant item’s rank, or any reported aggregate
metric. The hardened implementation passed the permutation-invariance checks.

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

`run_integrity_manifest.json` hashes retained run inputs and outputs.
`replay_manifest.json` hashes the frozen replay inputs and verifies replayed metrics
against the recorded run. Floating-point comparisons use the declared tolerance; the
packaged Amazon replay produced a maximum stored-output difference of
`1.1102230246251565e-16`, below the `1e-12` verification threshold.

## Repository map

```text
configs/paper.yaml                    accepted-submission parameters and 20-run check
configs/camera_ready.yaml             camera-ready 100-seed analysis parameters
src/tie_eval/                         data, scoring, ranking, metrics, diagnostics
scripts/download_data.py              official-source downloads and checksums
scripts/run_movielens.py               full MovieLens regeneration
scripts/run_bpc.py                     staged Amazon regeneration
scripts/replay_frozen.py               evaluator-only replay from saved rows/scores
scripts/make_paper_outputs.py          camera-ready Tables 2 and 3 data products
scripts/verify_artifact.py             smoke, archive, and full verification
results/archived/                      accepted aggregate evidence and provenance
results/camera_ready/                  compact camera-ready Amazon summaries
results/regenerated/                   locally generated outputs; not distributed
docs/CAMERA_READY_PROVENANCE.md        historical/reconstructed/new evidence boundary
examples/toy/                          data-free mechanism example
tests/                                 unit and toy end-to-end tests
```

## Resource guidance

| Run | RAM | Free disk | Typical CPU time |
|---|---:|---:|---:|
| tests and archived verification | <2 GB | <1 GB | under a few minutes |
| MovieLens full regeneration | 12–24 GB | 5–10 GB | roughly 10–40 minutes |
| Amazon raw/centered paper-scale audit | 32–64 GB | 40–80 GB | several hours |
| Amazon with cross-fitted group control | 64–96 GB recommended | 60–100 GB | several hours to more than one day |

Runtime depends on storage and CPU throughput. Amazon downloads, staged Parquet files,
processed splits, propensity outputs, control predictions, and score artifacts are
reused unless `--force` is supplied.

Start with `docs/REVIEWER_GUIDE.md`. The evidence boundary for the final paper is in
`docs/CAMERA_READY_PROVENANCE.md`; broader limitations are in
`docs/KNOWN_LIMITATIONS.md`.
