# Tie Handling Is Part of the Evaluation Protocol

Reviewer artifact for an order-invariance audit of tie-heavy recommender scores.

This repository is deliberately small. It separates three things that are easy to blur together:

1. an exactly reproducible evaluator mechanism on frozen or toy rows;
2. the archived float32-key policy used for the manuscript's reported aggregates;
3. a collision-hardened uint64 policy that satisfies the manuscript's normative row-order-invariance requirement.

The repository also preserves the reported aggregate results in machine-readable form. It does **not** pretend that aggregate CSVs are a substitute for the missing frozen candidate rows. The exact evidence boundary is documented in [EVIDENCE_STATUS.md](EVIDENCE_STATUS.md).

## Reviewer quick start

With Python 3.10 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[test]"
python scripts/toy_audit.py
python scripts/verify_reported_results.py
python -m pytest -q
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead of the POSIX activation command.

The audit commands do not access the network or an external dataset. The installation step may download NumPy, pytest, setuptools, or wheel when they are not already available in the reviewer’s environment. The audits verify the analytic all-tied example, compare stable/archived/hardened/randomized policies, check the arithmetic in the reported result tables, and run the invariance regression suite.

## What the artifact establishes

For one relevant item and 30 negatives with identical scores:

| Tie treatment | Hit@10 | NDCG@10 |
|---|---:|---:|
| Stable, positive first | 1.0000 | 1.0000 |
| Stable, positive last | 0.0000 | 0.0000 |
| Uniform tie-order expectation | 0.3226 | 0.1466 |

The candidate identities, labels, and primary scores are unchanged. Only the secondary ordering changes. This is the minimal counterexample behind the paper.

The empirical aggregates reported in the manuscript are available under `results/`. Their status differs by dataset:

- **BPC:** the archive contains aggregate stable/keyed values and tie diagnostics, but not the frozen candidate rows or the original tie-experiment output directory.
- **MovieLens:** the manuscript reports a historical paired stable/keyed run. The supplied archive retains a later keyed-only snapshot whose values differ slightly and no row-level diagnostic arrays.

Consequently, the mechanism and evaluator properties are executable here; the manuscript aggregates are inspectable and arithmetic-checked; full row-level empirical reproduction still requires restoring or regenerating the frozen rows described in [FROZEN_ROW_SCHEMA.md](FROZEN_ROW_SCHEMA.md).

## Tie policies

`stable`
: Descending primary score with input position preserved inside exact ties. This is intentionally order-dependent and serves as the failure-mode baseline.

`archived_float32`
: Reproduces the manuscript-described and retained BPC implementation: mix `(seed, user_id, item_id)` in wrapping uint64 arithmetic, divide by `2^64`, cast to float32, sort by the secondary key, then stably by descending primary score. It normally removes position bias, but float32 collisions can reintroduce input-layout dependence. The retained MovieLens script uses `lexsort`; the forms agree when secondary keys are unique but need not share collision behavior.

`hardened_uint64`
: Retains the full uint64 mixed key and uses item identity as a final fallback. This is the recommended position-independent operational policy.

`randomized`
: Draws independent secondary keys and orders exact ties without perturbing primary scores. Repeating this policy estimates the uniform tie-order expectation.

The deterministic keyed result is one reproducible realization inside each tie block. It is not the expectation over all admissible tie orders.

## Repository map

```text
configs/                    Reconstructed protocol records and explicit unknowns
results/                    Paper-reported aggregates and locally retained snapshots
scripts/toy_audit.py        Network-free executable demonstration
scripts/audit_frozen_rows.py Generic audit for a frozen-row NPZ
scripts/verify_reported_results.py
scripts/check_anonymity.py
scripts/build_manifest.py
src/tie_audit/              Archived and hardened evaluator implementations
tests/                      Analytic, invariance, collision, and regression tests
```

For the fastest review path, read [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md). For reconstruction levels and data boundaries, see [REPRODUCING.md](REPRODUCING.md).

The exact verification environment and passed gates are recorded in [TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md).

## Scope and non-claims

- The audit is conditional on the declared candidate universe. It does not claim that sampled and full-catalog evaluation are equivalent.
- A keyed total order provides reproducibility, not marginalization over ties.
- The two empirical domains demonstrate a failure condition; they do not estimate its prevalence across all recommender systems.
- No raw Amazon or MovieLens data are redistributed here.
- Apart from the repository's MIT license and hosting metadata, the research artifact contains no author affiliations, personal URLs, original development history, or development logs.

The archived BPC sampler could emit duplicate negatives or short rows after an attempt cap. The historical counts are unavailable because the frozen rows were not retained. The hardened NPZ contract rejects duplicate candidate identities; that is an improved audit requirement, not a claim of byte-equivalence with every historical row.

## Venue alignment

The paper is an evaluation-methodology contribution and aligns most directly with FRAME 2026's **Experimental Methodologies** track: protocol design, methodological critique, and recommendations for comparable empirical work. That track is single-blind and allows complete results; the eight-page manuscript body plus references matches its stated `8 pages + references` limit. See the official [FRAME call for contributions](https://remaplab.github.io/frame2026/cfp/) for the governing rules.

The generic RecSys call for proposals to *organize* a workshop is not the governing call for this paper.
