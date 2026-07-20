from __future__ import annotations

import numpy as np
import pandas as pd

from tie_eval.sampling import (
    SortedUserHistory,
    sample_bpc_paper_compatible,
    sample_unique_portable,
)


def _history() -> SortedUserHistory:
    return SortedUserHistory(
        np.array([0, 0, 1, 1], dtype=np.int64),
        np.array([1, 2, 3, 4], dtype=np.int64),
        n_users=2,
    )


def test_paper_sampler_is_deterministic_positive_first_and_history_safe() -> None:
    target = pd.DataFrame({"user_idx": [0, 1], "item_idx": [3, 5]})
    first = sample_bpc_paper_compatible(
        target, _history(), n_items=20, num_negatives=8, seed=42
    )
    second = sample_bpc_paper_compatible(
        target, _history(), n_items=20, num_negatives=8, seed=42
    )
    np.testing.assert_array_equal(first.candidates, second.candidates)
    np.testing.assert_array_equal(first.candidates[:, 0], target["item_idx"].to_numpy())
    for row, user in enumerate(target["user_idx"]):
        seen = set(_history().items_for_user(int(user)).tolist())
        assert not any(int(item) in seen for item in first.candidates[row, 1:])
        assert int(target.iloc[row]["item_idx"]) not in set(first.candidates[row, 1:].tolist())


def test_unique_sampler_has_no_duplicate_negatives() -> None:
    target = pd.DataFrame({"user_idx": [0, 1], "item_idx": [3, 5]})
    rows = sample_unique_portable(
        target, _history(), n_items=30, num_negatives=10, seed=42
    )
    for row in rows.candidates:
        assert len(set(row[1:].tolist())) == len(row[1:])
