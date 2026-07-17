# Tested Environment

The reviewer artifact was verified on Windows 10 (`10.0.26200`) with:

| Component | Version |
|---|---:|
| Python | 3.11.9 |
| NumPy | 2.4.1 |
| pytest | 9.0.3 |

An offline editable-package build was also verified with Python 3.12.13, setuptools 82.0.1, wheel 0.47.0, and NumPy 2.3.5.

Validation gates passed on the final source tree:

- 11 evaluator/unit/invariance/CLI-safety tests;
- all-tied analytic and Monte Carlo toy audit;
- manuscript aggregate and provenance arithmetic checks;
- end-to-end frozen-row NPZ smoke audit, including hashes, diagnostics, collision counts, and permutation comparisons;
- relative Markdown link check;
- strict anonymity scan;
- editable package build from `pyproject.toml`.

No claim is made that the historical BPC or MovieLens empirical rows were rerun in this environment. Those rows are absent from the supplied calculation archive, as documented in [EVIDENCE_STATUS.md](EVIDENCE_STATUS.md).
