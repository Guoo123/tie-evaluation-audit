from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CandidateRows:
    user_indices: np.ndarray
    positive_items: np.ndarray
    candidates: np.ndarray
    candidate_counts: np.ndarray


class SortedUserHistory:
    """Compact user-history membership structure for large datasets."""

    def __init__(self, user_indices: np.ndarray, item_indices: np.ndarray, n_users: int):
        pairs = np.column_stack(
            [np.asarray(user_indices, dtype=np.int64), np.asarray(item_indices, dtype=np.int64)]
        )
        pairs = np.unique(pairs, axis=0)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
        counts = np.bincount(pairs[:, 0], minlength=n_users)
        self.offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
        self.items = pairs[:, 1].astype(np.int64, copy=False)

    def contains(self, user: int, item: int) -> bool:
        start, end = int(self.offsets[user]), int(self.offsets[user + 1])
        row = self.items[start:end]
        position = int(np.searchsorted(row, item))
        return position < len(row) and int(row[position]) == int(item)

    def items_for_user(self, user: int) -> np.ndarray:
        start, end = int(self.offsets[user]), int(self.offsets[user + 1])
        return self.items[start:end]


def history_from_frames(frames: list[pd.DataFrame], n_users: int) -> SortedUserHistory:
    users = np.concatenate([frame["user_idx"].to_numpy(dtype=np.int64) for frame in frames])
    items = np.concatenate([frame["item_idx"].to_numpy(dtype=np.int64) for frame in frames])
    return SortedUserHistory(users, items, n_users)


def sample_bpc_paper_compatible(
    target_df: pd.DataFrame,
    history: SortedUserHistory,
    *,
    n_items: int,
    num_negatives: int,
    seed: int,
) -> CandidateRows:
    """Reproduce the archived BPC rejection sampler.

    The sampler intentionally does not reject duplicate negatives because the
    historical implementation did not.  This behavior is documented and isolated
    under an explicit paper-compatible mode.
    """
    rng = np.random.default_rng(seed)
    users = target_df["user_idx"].to_numpy(dtype=np.int64)
    positives = target_df["item_idx"].to_numpy(dtype=np.int64)
    rows = len(target_df)
    candidates = np.zeros((rows, num_negatives + 1), dtype=np.int64)
    candidates[:, 0] = positives
    counts = np.ones(rows, dtype=np.int32)

    for row, (user, positive) in enumerate(zip(users, positives)):
        negatives: list[int] = []
        attempts = 0
        max_attempts = num_negatives * 5
        while len(negatives) < num_negatives and attempts < max_attempts:
            batch = rng.integers(0, n_items, size=min(num_negatives * 2, 1000))
            for candidate in batch:
                candidate_int = int(candidate)
                if candidate_int != int(positive) and not history.contains(int(user), candidate_int):
                    negatives.append(candidate_int)
                    if len(negatives) >= num_negatives:
                        break
            attempts += 1
        obtained = len(negatives)
        if obtained:
            candidates[row, 1 : 1 + obtained] = negatives[:obtained]
        counts[row] = 1 + obtained
    return CandidateRows(users, positives, candidates, counts)


def sample_unique_portable(
    target_df: pd.DataFrame,
    history: SortedUserHistory,
    *,
    n_items: int,
    num_negatives: int,
    seed: int,
) -> CandidateRows:
    """Portable hardened sampler with unique negatives."""
    rng = np.random.default_rng(seed)
    users = target_df["user_idx"].to_numpy(dtype=np.int64)
    positives = target_df["item_idx"].to_numpy(dtype=np.int64)
    candidates = np.empty((len(target_df), num_negatives + 1), dtype=np.int64)
    candidates[:, 0] = positives
    counts = np.full(len(target_df), num_negatives + 1, dtype=np.int32)
    all_items = np.arange(n_items, dtype=np.int64)

    for row, (user, positive) in enumerate(zip(users, positives)):
        excluded = np.union1d(history.items_for_user(int(user)), np.array([positive], dtype=np.int64))
        eligible = np.setdiff1d(all_items, excluded, assume_unique=True)
        if len(eligible) < num_negatives:
            raise ValueError(f"user {user} has only {len(eligible)} eligible negatives")
        candidates[row, 1:] = rng.choice(eligible, size=num_negatives, replace=False)
    return CandidateRows(users, positives, candidates, counts)
