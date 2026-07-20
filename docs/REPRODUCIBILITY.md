# Reproducibility model

This artifact uses four evidence levels rather than treating all forms of
“reproducibility” as interchangeable.

## Level 1 — Mechanism verification

No external data is needed. Unit tests prove that a stable positive-first row can
change top-k metrics under exact ties, that the analytic formulas match direct
enumeration, and that the hardened policy is invariant to candidate-row permutation.

## Level 2 — Archived aggregate verification

`results/archived/` contains the aggregate values and configuration/provenance files
retained from the development ZIP. These files support arithmetic and claim-to-output
checks. They do not support row-by-row replay where candidate rows were absent.

## Level 3 — End-to-end regeneration

A new run starts from official public data and recreates:

1. raw data acquisition and integrity manifest;
2. preprocessing and split;
3. labels and item representations;
4. user score construction;
5. candidate sampling;
6. frozen candidate and score rows;
7. all tie policies and diagnostics; and
8. paper tables and figure data.

The expected result is the same qualitative mechanism and numerically close aggregate
metrics. A static upstream archive plus a pinned environment may reproduce the values
exactly, but exact floating-point identity is not promised across BLAS libraries,
operating systems, or changed upstream files.

## Level 4 — Prospective replay of a regenerated run

After regeneration, the artifact saves frozen candidate rows and score construction
artifacts. The evaluator can then be replayed without downloading or resampling.
This is stronger than the historical archive and is the recommended preservation
standard for any camera-ready release.

## Determinism controls

The declared random inputs are:

- global/preprocessing seed: 42;
- BPC negative-sampling seed: 42;
- MovieLens negative-sampling seed: 42;
- BPC archived/hardened key seed: 20260316;
- MovieLens archived/hardened key seed: 20260318;
- repeated-randomized seed sequence: configured in `configs/paper.yaml`.

The candidate-sampling seed and tie-key seed serve different roles. The first defines
the candidate universe. The second selects an order among candidates with equal
primary scores.

## Numerical comparison policy

The verification tolerances are declared, not hidden:

- MovieLens aggregate metrics: absolute tolerance 0.0005;
- BPC aggregate metrics: absolute tolerance 0.002;
- processed dataset counts: exact equality.

A tolerance failure produces a nonzero exit status. The generated values remain on
disk for diagnosis.

## Output integrity

Every full run writes:

- `runtime_manifest.json`, including Python, platform, and package versions;
- a data download manifest with observed hashes;
- `run_summary.json`, including processed counts and runtime;
- `frozen_candidate_rows.npz` when enabled;
- `metrics_by_policy.csv`;
- `diagnostics.json`; and
- repeated-randomized seed-level CSV files.

The file layout is deterministic and suitable for later checksumming and archival.
