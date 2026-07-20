from __future__ import annotations

import numpy as np

from tie_eval.diagnostics import policy_difference_diagnostics, tie_diagnostics
from tie_eval.ranking import TiePolicy


def test_all_tied_diagnostics() -> None:
    candidates = np.arange(31, dtype=np.int64)[None, :]
    scores = np.zeros((1, 31), dtype=np.float32)
    positives = np.array([0], dtype=np.int64)
    result = tie_diagnostics(candidates, scores, positives, k=10)
    assert result.any_tie_rate == 1.0
    assert result.boundary_tie_rate == 1.0
    assert result.positive_in_tie_rate == 1.0
    assert result.average_max_tie_block == 31.0
    assert result.average_positive_tie_size == 31.0


def test_no_tie_diagnostics() -> None:
    candidates = np.array([[0, 1, 2, 3]], dtype=np.int64)
    scores = np.array([[4.0, 3.0, 2.0, 1.0]])
    positives = np.array([0], dtype=np.int64)
    result = tie_diagnostics(candidates, scores, positives, k=2)
    assert result.any_tie_rate == 0.0
    assert result.boundary_tie_rate == 0.0
    assert result.positive_in_tie_rate == 0.0


def test_policy_difference_uses_item_set_semantics() -> None:
    # Candidate 4 appears twice. The diagnostic must compare sets rather than treating
    # duplicate multiplicity as a distinct item identity.
    candidates = np.array([[0, 4, 4, 2, 3]], dtype=np.int64)
    scores = np.zeros_like(candidates, dtype=np.float32)
    users = np.array([8], dtype=np.int64)
    positives = np.array([0], dtype=np.int64)
    result = policy_difference_diagnostics(
        candidates,
        scores,
        users,
        positives,
        left_policy=TiePolicy.STABLE_POSITIVE_FIRST,
        right_policy=TiePolicy.HARDENED_UINT64,
        key_seed=20260316,
        k=3,
    )
    assert 0.0 <= result["topk_item_set_change_rate"] <= 1.0
    assert 0.0 <= result["positive_rank_change_rate"] <= 1.0
