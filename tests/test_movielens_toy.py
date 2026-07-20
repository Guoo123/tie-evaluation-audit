from __future__ import annotations

import numpy as np
import pandas as pd

from tie_eval.movielens import compute_user_scores, sample_candidates_paper_compatible, score_candidate_rows


def test_movielens_score_and_sampler_components() -> None:
    train = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1],
            "item_idx": [0, 1, 1, 2],
        }
    )
    label_matrix = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )
    sources = compute_user_scores(train, label_matrix, n_users=2)
    np.testing.assert_array_equal(sources["raw_count"][0], np.array([2, 1], dtype=np.float32))

    test = pd.DataFrame({"user_idx": [0, 1], "item_idx": [2, 0]})
    users, positives, candidates = sample_candidates_paper_compatible(
        train, test, n_items=6, num_negatives=2, seed=42
    )
    np.testing.assert_array_equal(candidates[:, 0], positives)
    scores = score_candidate_rows(
        sources["raw_count"], label_matrix, users, candidates, batch_rows=1
    )
    assert scores.shape == candidates.shape
