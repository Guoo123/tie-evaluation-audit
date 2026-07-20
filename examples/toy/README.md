# All-tied mechanism example

Run:

```bash
python examples/toy/run_toy.py
```

The row contains one positive in column zero followed by 30 negatives. Every score is
zero. Stable sorting therefore gives the positive rank 1, whereas the analytic metric
averages its rank uniformly over all 31 admissible positions. The hardened policy is a
single reproducible position-independent realization, not the analytic expectation.
