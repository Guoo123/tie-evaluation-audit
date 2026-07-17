# Reviewer Guide

This artifact is organized around questions a skeptical reviewer should be able to answer quickly.

## 1. Is the failure mechanical?

Run:

```bash
python scripts/toy_audit.py
```

The script freezes one positive, 30 negatives, all item identities, all labels, and all scores. It then changes only the treatment of exact ties. It also prints the analytic expectation over all admissible positions of the positive.

## 2. Does the proposed policy ignore row storage order?

Run:

```bash
pytest -q
```

The tests permute candidate rows and verify ranking-level invariance for `hardened_uint64`, plus equivalence with ordinary score sorting when no ties exist. They separately preserve `archived_float32` for exact reproduction and test its collision behavior rather than silently treating it as a perfect implementation of the normative property.

## 3. Are the paper numbers machine-readable and internally consistent?

Run:

```bash
python scripts/verify_reported_results.py
```

Then inspect:

- `results/manuscript_table2.csv`
- `results/manuscript_bpc_tie_diagnostics.csv`
- `results/local_bpc_tie_uncertainty_snapshot.csv`
- `results/local_movielens_keyed_snapshot.csv`
- `results/movielens_manuscript_vs_local.csv`

The verifier checks full stored precision where available and verifies that displayed changes agree with stable and keyed values at the manuscript's rounding precision.

## 4. Can the empirical rows be rerun from this export alone?

No. That boundary is intentional and visible. The supplied development archive omitted the BPC frozen candidate/score rows and the historical MovieLens paired run. The retained aggregates are evidence of recorded calculations, not a full row-level reproduction package.

[EVIDENCE_STATUS.md](EVIDENCE_STATUS.md) maps each paper claim to what is present, what is executable, and what remains missing. [FROZEN_ROW_SCHEMA.md](FROZEN_ROW_SCHEMA.md) defines the exact portable payload needed to close the gap without publishing raw user histories.

## 5. What should be used in future benchmarks?

Use `hardened_uint64` when a single reproducible total order is the intended estimand. Use analytic expectation or repeated randomized secondary keys when the intended estimand is the metric implied only by the primary scores. Always report the candidate universe, row construction, tie prevalence, tie treatment, a permutation audit, and sensitivity to another admissible tie policy.

## Claim-to-artifact map

| Claim | Primary artifact | Verification |
|---|---|---|
| Stable positive-first rows can inflate top-k metrics | `scripts/toy_audit.py` | Toy run + analytic formulas |
| Archived policy is exactly preserved | `src/tie_audit/evaluator.py` | Golden/determinism tests |
| Hardened policy is row-order invariant | `tests/test_evaluator.py` | Candidate permutation tests |
| Keyed order is not tie marginalization | `scripts/toy_audit.py` | Keyed realization vs analytic/randomized mean |
| BPC aggregate change and tie rates | `results/` | Arithmetic verifier; aggregate-only provenance |
| MovieLens cross-domain aggregate change | `results/` | Arithmetic verifier; historical-source limitation |
