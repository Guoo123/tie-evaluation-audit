from __future__ import annotations

import numpy as np

from tie_eval.diagnostics import policy_difference_diagnostics, tie_diagnostics
from tie_eval.metrics import evaluate_policy, evaluate_policy_batched
from tie_eval.ranking import TiePolicy


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    rows, width = 37, 11
    candidates = np.stack([rng.permutation(width) for _ in range(rows)]).astype(np.int64)
    positives = candidates[:, 0].copy()
    scores = rng.integers(0, 4, size=(rows, width)).astype(np.float32)
    users = np.arange(rows, dtype=np.int64) + 100
    return candidates, scores, users, positives


def test_batched_deterministic_metrics_match_full_matrix() -> None:
    candidates, scores, users, positives = _data()
    for policy in (
        TiePolicy.STABLE_POSITIVE_FIRST,
        TiePolicy.ARCHIVED_FLOAT32,
        TiePolicy.HARDENED_UINT64,
    ):
        direct = evaluate_policy(
            candidates,
            scores,
            users,
            positives,
            policy=policy,
            key_seed=20260316,
            k=5,
        )
        batched = evaluate_policy_batched(
            candidates,
            scores,
            users,
            positives,
            policy=policy,
            key_seed=20260316,
            k=5,
            batch_rows=7,
        )
        assert direct.n_rows == batched.n_rows
        assert abs(direct.hit_at_k - batched.hit_at_k) < 1e-15
        assert abs(direct.ndcg_at_k - batched.ndcg_at_k) < 1e-15
        assert abs(direct.mrr - batched.mrr) < 1e-15


def test_randomized_metrics_do_not_depend_on_batch_size() -> None:
    candidates, scores, users, positives = _data()
    small = evaluate_policy_batched(
        candidates,
        scores,
        users,
        positives,
        policy=TiePolicy.RANDOMIZED,
        key_seed=91,
        random_seed=91,
        k=5,
        batch_rows=3,
    )
    large = evaluate_policy_batched(
        candidates,
        scores,
        users,
        positives,
        policy=TiePolicy.RANDOMIZED,
        key_seed=91,
        random_seed=91,
        k=5,
        batch_rows=100,
    )
    assert small.n_rows == large.n_rows
    assert abs(small.hit_at_k - large.hit_at_k) < 1e-15
    assert abs(small.ndcg_at_k - large.ndcg_at_k) < 1e-15
    assert abs(small.mrr - large.mrr) < 1e-15


def test_batched_diagnostics_are_batch_size_invariant() -> None:
    candidates, scores, users, positives = _data()
    tie_small = tie_diagnostics(candidates, scores, positives, k=5, batch_rows=4)
    tie_large = tie_diagnostics(candidates, scores, positives, k=5, batch_rows=100)
    assert tie_small == tie_large

    diff_small = policy_difference_diagnostics(
        candidates,
        scores,
        users,
        positives,
        left_policy=TiePolicy.STABLE_POSITIVE_FIRST,
        right_policy=TiePolicy.HARDENED_UINT64,
        key_seed=20260316,
        k=5,
        batch_rows=4,
    )
    diff_large = policy_difference_diagnostics(
        candidates,
        scores,
        users,
        positives,
        left_policy=TiePolicy.STABLE_POSITIVE_FIRST,
        right_policy=TiePolicy.HARDENED_UINT64,
        key_seed=20260316,
        k=5,
        batch_rows=100,
    )
    assert diff_small == diff_large
