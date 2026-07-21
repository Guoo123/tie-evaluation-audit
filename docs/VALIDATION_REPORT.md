# Artifact validation report

This report records checks performed while assembling the focused distribution.
It distinguishes code/package validation from full public-data regeneration.

## Validation scope

The construction environment verified:

- the all-tied theoretical example and analytic expectations;
- stable, archived-float32, hardened-uint64, and randomized ranking policies;
- row-permutation invariance of the hardened policy;
- no-tie equivalence across tie policies;
- agreement between full-matrix and bounded-memory evaluators;
- exact-score tie diagnostics and policy-difference diagnostics;
- BPC and MovieLens toy end-to-end pipelines;
- retained aggregate evidence, dataset counts, table arithmetic, and the
  machine-readable claim-to-evidence map;
- reconstruction of Table 2 and Figure 1 from explicitly labeled archived evidence;
- package-hygiene scanning; and
- SHA-256 integrity of all distributable files.

## Commands and outcomes

The following commands were executed from the repository root:

```bash
python -m compileall -q src scripts tests examples
PYTHONPATH=src pytest -q
PYTHONPATH=src python examples/toy/run_toy.py
PYTHONPATH=src python scripts/make_paper_outputs.py --config configs/paper.yaml
python scripts/verify_manifest.py
PYTHONPATH=src python scripts/verify_artifact.py --level archive
python scripts/check_package_hygiene.py
```

Final outcomes:

| Check | Outcome |
|---|---|
| Test suite in the Python 3.13 compatibility audit | 23 tests passed |
| All-tied stable Hit@10 / NDCG@10 | 1.0 / 1.0 |
| All-tied analytic Hit@10 | 10/31 = 0.3225806452 |
| All-tied analytic NDCG@10 | 0.1465664303 |
| Paper-output source in this environment | archived aggregate evidence, explicitly labeled |
| Archived paper-table rows checked | 8 |
| Claim-evidence entries required by verifier | present |
| Distribution manifest | 81 files verified |
| Package-hygiene scan | passed |

The generated paper-output manifest recorded eight archived rows and zero freshly
regenerated full-data rows. Generated files under `results/regenerated/` were removed
before packaging; reviewers can recreate them with `scripts/make_paper_outputs.py`.
The artifact verifier now prints its report without modifying the checkout unless an
explicit `--output` path is supplied.

## Environment used for package construction

The available construction runtime was Python 3.13.5 rather than the paper-pinned
Python 3.11 environment. Core tests also ran with several package versions newer than
the pinned environment. This mismatch was detected by `scripts/preflight.py` and is
not presented as a paper-compatible numerical environment.

The exact paper environment remains Python 3.11 with the pinned versions in
`requirements.txt`, `environment.yml`, the Dockerfile, and CI. `pyproject.toml` now
permits compatible Python 3.11--3.13 environments with bounded dependency ranges for
low-friction reviewer inspection. The preflight command still identifies whether the
exact frozen environment is in use and exits nonzero under `--strict` otherwise.

## Full-data boundary

No fresh MovieLens 25M or Amazon Beauty & Personal Care numerical run was completed
in the construction environment. External public-data downloads were unavailable,
and the available runtime had only 4 GiB RAM, below the documented requirements.
Accordingly:

- archived files verify the retained submitted aggregates and their arithmetic;
- unit and toy end-to-end tests verify the code paths and evaluator mechanism;
- the public-data pipeline is provided for reviewer-side regeneration; and
- this report does **not** claim that the submitted historical candidate rows were
  replayed or that fresh full-data metrics were observed during packaging.

A completed reviewer run creates frozen candidate rows, candidate-score matrices,
resolved configuration, runtime metadata, download hashes, run-output hashes, and a
separate replay manifest. Those prospective artifacts support exact evaluator replay
of the new run even though the historical ZIP did not preserve its old rows.
