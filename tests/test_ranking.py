from __future__ import annotations

import numpy as np

from tie_eval.hashing import archived_float32_keys, mixed_uint64_keys
from tie_eval.ranking import TiePolicy, rank_candidates


def test_all_tied_stable_preserves_input_order() -> None:
    candidates = np.array([[9, 4, 7, 2]], dtype=np.int64)
    scores = np.zeros_like(candidates, dtype=np.float32)
    users = np.array([3], dtype=np.int64)
    ranked = rank_candidates(
        candidates, scores, users, TiePolicy.STABLE_POSITIVE_FIRST, seed=17
    )
    np.testing.assert_array_equal(ranked, candidates)


def test_hardened_ranking_is_invariant_to_row_permutation() -> None:
    rng = np.random.default_rng(123)
    rows = 100
    width = 31
    candidates = np.tile(np.arange(width, dtype=np.int64), (rows, 1))
    # Deliberately create several large exact tie blocks.
    scores = rng.integers(0, 4, size=(rows, width)).astype(np.float32)
    users = np.arange(rows, dtype=np.int64) + 1000
    reference = rank_candidates(
        candidates, scores, users, TiePolicy.HARDENED_UINT64, seed=20260316
    )

    permutations = np.stack([rng.permutation(width) for _ in range(rows)])
    permuted_candidates = np.take_along_axis(candidates, permutations, axis=1)
    permuted_scores = np.take_along_axis(scores, permutations, axis=1)
    reranked = rank_candidates(
        permuted_candidates,
        permuted_scores,
        users,
        TiePolicy.HARDENED_UINT64,
        seed=20260316,
    )
    np.testing.assert_array_equal(reranked, reference)


def test_all_policies_agree_when_primary_scores_are_strict() -> None:
    rng = np.random.default_rng(7)
    rows = 25
    width = 31
    candidates = np.stack([rng.permutation(width) for _ in range(rows)]).astype(np.int64)
    # The score follows item identity, so it remains attached to the same item after
    # any row representation change and has no exact ties.
    scores = candidates.astype(np.float64)
    users = np.arange(rows, dtype=np.int64)
    expected = rank_candidates(
        candidates, scores, users, TiePolicy.STABLE_POSITIVE_FIRST, seed=1
    )
    for policy in (TiePolicy.ARCHIVED_FLOAT32, TiePolicy.HARDENED_UINT64):
        actual = rank_candidates(candidates, scores, users, policy, seed=20260316)
        np.testing.assert_array_equal(actual, expected)


def test_archived_and_hardened_key_types_and_shapes() -> None:
    candidates = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    users = np.array([10, 11], dtype=np.int64)
    archived = archived_float32_keys(candidates, users, 20260316)
    hardened = mixed_uint64_keys(candidates, users, 20260316)
    assert archived.shape == candidates.shape
    assert hardened.shape == candidates.shape
    assert archived.dtype == np.float32
    assert hardened.dtype == np.uint64
    assert np.all((archived >= 0.0) & (archived <= 1.0))
