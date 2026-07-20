from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tie_eval.movielens import run
from tie_eval.replay import replay_dataset


def _write_toy_movielens(root: Path) -> Path:
    data_dir = root / "data/raw/movielens/ml-25m"
    data_dir.mkdir(parents=True)

    rows: list[dict[str, float | int]] = []
    timestamp = 1
    # 24 global-training rows. All six movies appear before the split.
    for repetition in range(4):
        for movie_id in range(1, 7):
            rows.append(
                {
                    "userId": 1 + (repetition % 3),
                    "movieId": movie_id,
                    "rating": 5.0,
                    "timestamp": timestamp,
                }
            )
            timestamp += 1
    # Three validation rows.
    for user_id, movie_id in ((1, 1), (2, 2), (3, 3)):
        rows.append(
            {
                "userId": user_id,
                "movieId": movie_id,
                "rating": 4.0,
                "timestamp": timestamp,
            }
        )
        timestamp += 1
    # Three test-only users, one row each.
    for user_id, movie_id in ((4, 1), (5, 2), (6, 3)):
        rows.append(
            {
                "userId": user_id,
                "movieId": movie_id,
                "rating": 5.0,
                "timestamp": timestamp,
            }
        )
        timestamp += 1
    pd.DataFrame(rows).to_csv(data_dir / "ratings.csv", index=False)

    pd.DataFrame(
        [
            {"tagId": 1, "tag": "action"},
            {"tagId": 2, "tag": "classic"},
        ]
    ).to_csv(data_dir / "genome-tags.csv", index=False)
    genome_rows = []
    for movie_id in range(1, 7):
        genome_rows.extend(
            [
                {
                    "movieId": movie_id,
                    "tagId": 1,
                    "relevance": 0.9 if movie_id % 2 else 0.1,
                },
                {
                    "movieId": movie_id,
                    "tagId": 2,
                    "relevance": 0.9 if movie_id <= 3 else 0.1,
                },
            ]
        )
    pd.DataFrame(genome_rows).to_csv(data_dir / "genome-scores.csv", index=False)
    return data_dir


def test_movielens_pipeline_and_frozen_replay(tmp_path: Path) -> None:
    data_dir = _write_toy_movielens(tmp_path)
    config_path = tmp_path / "configs/toy.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("artifact: toy\n", encoding="utf-8")
    config = {
        "_repo_root": str(tmp_path),
        "_config_path": str(config_path),
        "artifact": {
            "output_root": "results/regenerated",
            "candidate_count": 3,
            "num_negatives": 2,
            "cutoff": 2,
            "batch_rows": 10,
        },
        "policies": {
            "movielens_key_seed": 20260318,
            "bpc_key_seed": 20260316,
            "randomized_repetitions": 2,
            "randomized_seed_start": 20270000,
        },
        "movielens": {
            "download": {"extract_dir": str(data_dir.relative_to(tmp_path))},
            "preprocessing": {
                "positive_rating_threshold": 4.0,
                "selected_tags": ["action", "classic"],
                "tag_threshold": 0.7,
                "global_train_fraction": 0.8,
                "global_validation_fraction": 0.1,
                "global_test_fraction": 0.1,
            },
            "sampling": {"seed": 42},
            "run": {"max_test_users": 3, "keep_score_matrices": False},
        },
    }

    summary = run(config)
    assert summary["dataset"]["n_train"] == 24
    assert summary["dataset"]["n_validation"] == 3
    assert summary["dataset"]["n_test"] == 3

    output = tmp_path / "results/regenerated/movielens"
    metrics = pd.read_csv(output / "metrics_by_policy.csv")
    assert len(metrics) == 12
    assert set(metrics["score"]) == {"raw_count", "centered_count", "popularity"}
    assert set(metrics["policy"]) == {
        "stable_positive_first",
        "archived_float32",
        "hardened_uint64",
        "analytic_uniform_tie_expectation",
    }
    frozen = np.load(output / "frozen_candidate_rows.npz")
    assert frozen["candidates"].shape == (3, 3)
    np.testing.assert_array_equal(frozen["candidates"][:, 0], frozen["positive_items"])
    assert (output / "candidate_scores_raw_count.npy").exists()
    assert (output / "resolved_config.json").exists()
    assert (output / "run_integrity_manifest.json").exists()

    manifest = replay_dataset(
        config,
        "movielens",
        include_randomized=False,
        include_diagnostics=True,
    )
    assert manifest["recorded_metric_comparison"]["status"] == "pass"
    assert json.loads((output / "replay/replay_manifest.json").read_text())["dataset"] == "movielens"
