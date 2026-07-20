from __future__ import annotations

from enum import Enum

import numpy as np

from .hashing import archived_float32_keys, mixed_uint64_keys


class TiePolicy(str, Enum):
    STABLE_POSITIVE_FIRST = "stable_positive_first"
    ARCHIVED_FLOAT32 = "archived_float32"
    HARDENED_UINT64 = "hardened_uint64"
    RANDOMIZED = "randomized"


def _validate(candidate_matrix: np.ndarray, score_matrix: np.ndarray, users: np.ndarray) -> None:
    if candidate_matrix.ndim != 2 or score_matrix.ndim != 2:
        raise ValueError("candidate_matrix and score_matrix must be two-dimensional")
    if candidate_matrix.shape != score_matrix.shape:
        raise ValueError("candidate_matrix and score_matrix must have the same shape")
    if users.ndim != 1 or len(users) != len(candidate_matrix):
        raise ValueError("users must contain one identity per row")
    if not np.all(np.isfinite(score_matrix)):
        raise ValueError("primary scores contain NaN or infinity")


def rank_indices(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    user_indices: np.ndarray,
    policy: str | TiePolicy,
    *,
    seed: int,
    random_seed: int | None = None,
) -> np.ndarray:
    """Return row-wise ranking indices under a declared tie policy.

    ``stable_positive_first`` reproduces the problematic evaluator: stable sorting
    by score alone retains the input row order.  In paper-compatible candidate rows,
    the held-out positive occupies column zero.

    ``archived_float32`` reproduces the exact historical keyed evaluator.

    ``hardened_uint64`` is the recommended reference policy.  It orders by
    ``(-score, uint64_key, item_id)`` and therefore remains invariant to row
    permutation, including the rare case of a secondary-key collision.
    """
    candidates = np.asarray(candidate_matrix, dtype=np.int64)
    scores = np.asarray(score_matrix)
    users = np.asarray(user_indices, dtype=np.int64)
    _validate(candidates, scores, users)
    policy = TiePolicy(policy)

    if policy is TiePolicy.STABLE_POSITIVE_FIRST:
        return np.argsort(-scores, axis=1, kind="stable")

    if policy is TiePolicy.ARCHIVED_FLOAT32:
        keys = archived_float32_keys(candidates, users, seed)
        key_order = np.argsort(keys, axis=1, kind="stable")
        scores_by_key = np.take_along_axis(scores, key_order, axis=1)
        score_order = np.argsort(-scores_by_key, axis=1, kind="stable")
        return np.take_along_axis(key_order, score_order, axis=1)

    if policy is TiePolicy.HARDENED_UINT64:
        keys = mixed_uint64_keys(candidates, users, seed)
        # np.lexsort uses the final key as primary.  item identity is a declared
        # collision fallback and can never reverse unequal primary scores.
        return np.lexsort((candidates, keys, -scores), axis=1)

    if random_seed is None:
        random_seed = seed
    rng = np.random.default_rng(random_seed)
    random_keys = rng.random(size=scores.shape, dtype=np.float64)
    return np.lexsort((candidates, random_keys, -scores), axis=1)


def rank_candidates(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    user_indices: np.ndarray,
    policy: str | TiePolicy,
    *,
    seed: int,
    random_seed: int | None = None,
) -> np.ndarray:
    order = rank_indices(
        candidate_matrix,
        score_matrix,
        user_indices,
        policy,
        seed=seed,
        random_seed=random_seed,
    )
    return np.take_along_axis(np.asarray(candidate_matrix), order, axis=1)
