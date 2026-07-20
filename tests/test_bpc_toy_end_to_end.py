from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")

from tie_eval.bpc import run
from tie_eval.config import load_config


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _toy_config(root: Path) -> Path:
    config = {
        "artifact": {
            "schema_version": 1,
            "output_root": "results/regenerated",
            "global_seed": 42,
            "candidate_count": 2,
            "num_negatives": 1,
            "cutoff": 1,
            "save_candidate_rows": True,
            "save_candidate_scores": True,
            "batch_rows": 100,
        },
        "policies": {
            "bpc_key_seed": 20260316,
            "movielens_key_seed": 20260318,
            "randomized_repetitions": 2,
            "randomized_seed_start": 700,
        },
        "bpc": {
            "download": {
                "reviews_path": "data/raw/bpc/reviews.jsonl.gz",
                "metadata_path": "data/raw/bpc/meta.jsonl.gz",
            },
            "preprocessing": {
                "min_user_interactions": 3,
                "min_item_interactions": 3,
                "max_kcore_iterations": 25,
            },
            "labels": {
                "names": ["fragrance_free", "cruelty_free", "sulfate_free"],
                "unknown_as_negative": True,
            },
            "propensity": {
                "folds": 2,
                "max_features": 100,
                "min_df": 1,
                "C": 1.0,
                "max_iter": 100,
                "class_weight": "balanced",
                "trim_above": 0.95,
                "random_seed": 42,
            },
            "group_control": {
                "folds": 2,
                "smoothing": 2.0,
                "min_user_group_count": 1,
                "min_item_group_support": 1,
                "include_store": True,
                "include_category": True,
                "include_price_bucket": True,
                "random_seed": 42,
            },
            "sampling": {
                "seed": 42,
                "mode": "paper_compatible_rejection",
            },
        },
    }
    path = root / "configs" / "toy.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_bpc_pipeline_regenerates_from_raw_toy_data(tmp_path: Path) -> None:
    item_texts = [
        "fragrance-free unscented calming lotion with botanical moisturizer",
        "cruelty-free leaping bunny facial cleanser with gentle foam",
        "sulfate-free no sulfates shampoo for daily hair cleansing",
        "fragrance-free cruelty-free sulfate-free complete beauty product",
        "scented perfume beauty product with floral fragrance and rich texture",
        "ordinary moisturizing beauty product with vitamins and smooth texture",
    ]
    metadata = []
    for item in range(6):
        metadata.append(
            {
                "parent_asin": f"I{item}",
                "title": item_texts[item],
                "store": f"Store{item % 2}",
                "main_category": "Beauty",
                "categories": ["Beauty", f"Category{item % 3}"],
                "features": ["long lasting", "dermatology product"],
                "description": [item_texts[item]],
                "details": {"formula": "cream"},
                "price": 10 + item,
            }
        )

    reviews = []
    timestamp = 1
    # Six users, five distinct cyclic items each. Every user and item survives 3-core;
    # each user has three train rows, one validation row, and one test row.
    for user in range(6):
        for offset in range(5):
            reviews.append(
                {
                    "user_id": f"U{user}",
                    "parent_asin": f"I{(user + offset) % 6}",
                    "rating": 4 + (offset % 2),
                    "timestamp": timestamp,
                }
            )
            timestamp += 1

    _write_jsonl_gz(tmp_path / "data/raw/bpc/reviews.jsonl.gz", reviews)
    _write_jsonl_gz(tmp_path / "data/raw/bpc/meta.jsonl.gz", metadata)
    config = load_config(_toy_config(tmp_path))
    summary = run(config, stage="all", include_low_tie=True)

    assert summary["paper_scale"] is True
    assert summary["dataset"]["n_users"] == 6
    assert summary["dataset"]["n_items"] == 6
    assert summary["dataset"]["n_train"] == 18
    assert summary["dataset"]["n_validation"] == 6
    assert summary["dataset"]["n_test"] == 6
    metrics_path = tmp_path / "results/regenerated/bpc/metrics_by_policy.csv"
    metrics = pd.read_csv(metrics_path)
    assert set(metrics["score"]) == {
        "raw_count",
        "centered_count",
        "item_residual_only",
        "residualized_group_control",
    }
    assert {
        "stable_positive_first",
        "archived_float32",
        "hardened_uint64",
        "analytic_uniform_tie_expectation",
    }.issubset(set(metrics["policy"]))
    assert (tmp_path / "results/regenerated/bpc/frozen_candidate_rows.npz").exists()
    assert (tmp_path / "results/regenerated/bpc/candidate_scores_raw_count.npy").exists()
    assert (tmp_path / "results/regenerated/bpc/resolved_config.json").exists()
    assert (tmp_path / "results/regenerated/bpc/run_integrity_manifest.json").exists()
