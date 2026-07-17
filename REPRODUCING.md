# Reproducing the Audit

Reproduction has three levels. Keeping them separate prevents an aggregate-result check from being mistaken for a row-level rerun.

## Level 1: mechanism and evaluator properties

This level is fully self-contained:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python scripts/toy_audit.py
python -m pytest -q
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of the POSIX activation command.

Expected runtime is seconds on a CPU. After the dependencies are installed, no GPU, dataset download, or external service is needed.

## Level 2: reported-aggregate integrity

```bash
python scripts/verify_reported_results.py
python scripts/build_manifest.py --check
```

This checks stored table arithmetic, required files, and SHA-256 integrity. It does not recreate primary scores or sampled candidate rows.

## Level 3: empirical row-level audit

Export each frozen evaluation as an NPZ matching [FROZEN_ROW_SCHEMA.md](FROZEN_ROW_SCHEMA.md), then run:

```bash
python scripts/audit_frozen_rows.py \
  --input private/frozen_rows/bpc_raw_count.npz \
  --dataset bpc \
  --k 10 \
  --archived-seed 20260316 \
  --output generated/bpc_raw_count_audit.json
```

For MovieLens use seed `20260318`. The audit compares stable, archived, hardened, analytic, and repeated-random policies; reports exact-tie, boundary-tie, positive-in-tie, collision, and permutation statistics; and records hashes of the input arrays.

The command refuses to replace an existing report unless `--force` is supplied, and it never permits `--output` to refer to the input NPZ. Randomized metrics consume a continuous seeded stream and are regression-tested to remain identical when `--chunk-size` changes.

The `private/` and `generated/` directories are ignored by Git. Before sharing row payloads, confirm that dataset terms permit redistribution. When redistribution is not allowed, publish cryptographic hashes and deterministic reconstruction instructions instead.

## Dataset reconstruction notes

### Amazon Beauty & Personal Care

- Dataset family: Amazon Reviews 2023, Beauty and Personal Care category.
- Filtering recorded in the supplied calculation snapshot: minimum three interactions for both users and items.
- Split: chronological within user with one validation and one test holdout.
- Candidate row: one held-out positive followed by 30 sampled negatives.
- Negative exclusions recorded by the manuscript: the user's pre-test history and the positive item.
- Candidate sampling seed: `42`.
- Archived tie seed: `20260316`.

The supplied archive does not retain the frozen candidate rows, item-attribute matrices, or every score matrix needed to regenerate the reported audit directly.

### MovieLens 25M + Tag Genome

- Official dataset: [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/).
- Positive event: rating at least four stars.
- Tag concepts: action, classic, visually appealing, based on a book, violence, and science fiction; threshold `0.7`.
- Candidate row: one held-out positive followed by 30 sampled negatives.
- Negative exclusions in retained code: the positive item and the user's training history.
- Candidate sampling seed: `42`.
- Archived tie seed: `20260318`.
- Historical audit size: 10,000 test users.

The development script globally time-sorts positive ratings, takes an 80/10/10 interaction split, selects the first 10,000 unique users encountered in the test portion, and then retains the first test interaction per selected user. These choices are reconstructed in `configs/movielens_audit.json`; they should be reported because they are part of the estimand.

The archived BPC negative sampler did not explicitly de-duplicate accepted negatives and could stop with a short row after an attempt cap. Because the historical candidate rows are absent, duplicate and short-row counts are unknown. The portable frozen-row schema requires unique candidates and complete rows so future audits have an unambiguous candidate universe.

## Environment capture for a new full run

For any regenerated empirical snapshot, add:

- Python and NumPy versions;
- operating system and CPU/GPU model;
- dataset file hashes;
- configuration hash;
- exact candidate-row NPZ hashes;
- wall-clock runtime;
- stable, archived, hardened, analytic, and randomized outputs;
- archived-key and hardened-key collision counts;
- full row-permutation comparison counts.

Do not replace an old aggregate silently. Add a new immutable run identifier and explain any difference.
