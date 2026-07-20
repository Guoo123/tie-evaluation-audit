from __future__ import annotations

import itertools

import numpy as np

from tie_eval.metrics import (
    aggregate_from_ranks,
    analytic_single_positive_expectation,
    evaluate_policy,
)
from tie_eval.ranking import TiePolicy


def test_paper_all_tied_example() -> None:
    candidates = np.arange(31, dtype=np.int64)[None, :]
    scores = np.zeros((1, 31), dtype=np.float32)
    users = np.array([0], dtype=np.int64)
    positives = np.array([0], dtype=np.int64)

    stable = evaluate_policy(
        candidates,
        scores,
        users,
        positives,
        policy=TiePolicy.STABLE_POSITIVE_FIRST,
        key_seed=20260316,
        k=10,
    )
    analytic = analytic_single_positive_expectation(scores, candidates, positives, k=10)

    assert stable.hit_at_k == 1.0
    assert stable.ndcg_at_k == 1.0
    np.testing.assert_allclose(analytic.hit_at_k, 10 / 31, rtol=0, atol=1e-12)
    expected_ndcg = sum(1 / np.log2(rank + 1) for rank in range(1, 11)) / 31
    np.testing.assert_allclose(analytic.ndcg_at_k, expected_ndcg, rtol=0, atol=1e-12)


def test_analytic_expectation_matches_exhaustive_permutations() -> None:
    # Two candidates score above the positive; the positive is in a tie of size 3.
    candidates = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    scores = np.array([[1.0, 3.0, 2.0, 1.0, 1.0]], dtype=np.float64)
    positives = np.array([0], dtype=np.int64)
    analytic = analytic_single_positive_expectation(scores, candidates, positives, k=4)

    tied = [0, 3, 4]
    ranks: list[int] = []
    for permutation in itertools.permutations(tied):
        ranked = [1, 2, *permutation]
        ranks.append(ranked.index(0) + 1)
    exhaustive = aggregate_from_ranks(np.asarray(ranks), k=4)
    np.testing.assert_allclose(analytic.hit_at_k, exhaustive.hit_at_k, atol=1e-12)
    np.testing.assert_allclose(analytic.ndcg_at_k, exhaustive.ndcg_at_k, atol=1e-12)
    np.testing.assert_allclose(analytic.mrr, exhaustive.mrr, atol=1e-12)


def test_analytic_requires_exactly_one_positive_occurrence() -> None:
    candidates = np.array([[1, 1, 2]], dtype=np.int64)
    scores = np.zeros((1, 3), dtype=np.float32)
    positives = np.array([1], dtype=np.int64)
    try:
        analytic_single_positive_expectation(scores, candidates, positives, k=2)
    except ValueError as exc:
        assert "exactly one positive occurrence" in str(exc)
    else:
        raise AssertionError("expected ValueError")
