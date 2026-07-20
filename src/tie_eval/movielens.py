from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import repo_path
from .diagnostics import policy_difference_diagnostics, tie_diagnostics
from .metrics import (
    analytic_single_positive_expectation,
    evaluate_policy_batched,
    repeated_randomized_expectation,
)
from .provenance import (
    runtime_manifest,
    write_json,
    write_resolved_config,
    write_run_integrity_manifest,
)
from .ranking import TiePolicy

LOGGER = logging.getLogger(__name__)


def load_and_preprocess(config: dict[str, Any]) -> dict[str, Any]:
    settings = config["movielens"]
    data_dir = repo_path(config, settings["download"]["extract_dir"])
    prep = settings["preprocessing"]

    ratings = pd.read_csv(data_dir / "ratings.csv")
    genome_tags = pd.read_csv(data_dir / "genome-tags.csv")
    genome_scores = pd.read_csv(data_dir / "genome-scores.csv")

    tag_name_to_id = dict(zip(genome_tags["tag"], genome_tags["tagId"]))
    selected_names: list[str] = []
    tag_ids: list[int] = []
    for name in prep["selected_tags"]:
        if name not in tag_name_to_id:
            raise KeyError(f"Tag Genome concept not found: {name}")
        selected_names.append(name)
        tag_ids.append(int(tag_name_to_id[name]))

    genome_sub = genome_scores[genome_scores["tagId"].isin(tag_ids)].copy()
    genome_movies = genome_sub["movieId"].unique()
    positive = ratings[
        (ratings["rating"] >= float(prep["positive_rating_threshold"]))
        & ratings["movieId"].isin(genome_movies)
    ].copy()

    # These first-appearance maps deliberately reproduce the archived script.
    unique_users = positive["userId"].unique()
    unique_items = positive["movieId"].unique()
    user_to_idx = {user: index for index, user in enumerate(unique_users)}
    item_to_idx = {item: index for index, item in enumerate(unique_items)}
    positive["user_idx"] = positive["userId"].map(user_to_idx).astype(np.int64)
    positive["item_idx"] = positive["movieId"].map(item_to_idx).astype(np.int64)

    n_users = len(unique_users)
    n_items = len(unique_items)
    label_matrix = np.zeros((n_items, len(tag_ids)), dtype=np.float32)
    for label_index, tag_id in enumerate(tag_ids):
        tag_scores = genome_sub[genome_sub["tagId"] == tag_id].set_index("movieId")["relevance"]
        for movie_id, item_index in item_to_idx.items():
            if float(tag_scores.get(movie_id, 0.0)) >= float(prep["tag_threshold"]):
                label_matrix[item_index, label_index] = 1.0

    # Keep pandas' archived default sort behavior; dependency versions are pinned.
    positive = positive.sort_values("timestamp")
    train_end = int(len(positive) * float(prep["global_train_fraction"]))
    validation_end = int(
        len(positive)
        * (float(prep["global_train_fraction"]) + float(prep["global_validation_fraction"]))
    )
    train = positive.iloc[:train_end].copy()
    validation = positive.iloc[train_end:validation_end].copy()
    test = positive.iloc[validation_end:].copy()

    maximum_users = int(settings["run"]["max_test_users"])
    test_users = test["user_idx"].unique()[:maximum_users]
    test = test[test["user_idx"].isin(test_users)]
    # groupby defaults to sort=True, matching the historical script.
    test = test.groupby("user_idx").first().reset_index()

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "n_users": n_users,
        "n_items": n_items,
        "label_matrix": label_matrix,
        "label_names": selected_names,
    }


def compute_user_scores(train: pd.DataFrame, label_matrix: np.ndarray, n_users: int) -> dict[str, np.ndarray]:
    users = train["user_idx"].to_numpy(dtype=np.int64)
    items = train["item_idx"].to_numpy(dtype=np.int64)
    raw = np.zeros((n_users, label_matrix.shape[1]), dtype=np.float64)
    label_mean = label_matrix.mean(axis=0, keepdims=True)
    centered_items = label_matrix - label_mean
    centered = np.zeros_like(raw)

    # Chunked np.add.at preserves the source row sequence while avoiding a large
    # 10-million-by-6 temporary array.
    chunk_size = 500_000
    for start in range(0, len(train), chunk_size):
        end = min(start + chunk_size, len(train))
        np.add.at(raw, users[start:end], label_matrix[items[start:end]])
        np.add.at(centered, users[start:end], centered_items[items[start:end]])

    popularity = np.bincount(items, minlength=label_matrix.shape[0]).astype(np.float32)
    return {
        "raw_count": raw.astype(np.float32),
        "centered_count": centered.astype(np.float32),
        "popularity": popularity,
    }


def sample_candidates_paper_compatible(
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_items: int,
    *,
    num_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the original MovieLens set-subtraction sampler.

    Integer set iteration is retained because it is part of the archived program.
    Python 3.11 is pinned in ``environment.yml``.  The generated candidate rows are
    saved so downstream policy comparisons never resample.
    """
    rng = np.random.default_rng(seed)
    history = train.groupby("user_idx")["item_idx"].apply(set).to_dict()
    all_items = set(range(n_items))
    users = test["user_idx"].to_numpy(dtype=np.int64)
    positives = test["item_idx"].to_numpy(dtype=np.int64)
    candidates = np.empty((len(test), num_negatives + 1), dtype=np.int64)
    candidates[:, 0] = positives
    for row, (user, positive) in enumerate(zip(users, positives)):
        eligible = all_items - history.get(int(user), set()) - {int(positive)}
        if len(eligible) < num_negatives:
            raise ValueError(f"user {user} has too few eligible negatives")
        candidates[row, 1:] = rng.choice(list(eligible), size=num_negatives, replace=False)
    return users, positives, candidates


def score_candidate_rows(
    user_scores: np.ndarray,
    label_matrix: np.ndarray,
    users: np.ndarray,
    candidates: np.ndarray,
    *,
    batch_rows: int = 1000,
) -> np.ndarray:
    """Score frozen rows using the same float32 matrix product as the archived run."""
    output = np.empty(candidates.shape, dtype=np.float32)
    for start in range(0, len(users), batch_rows):
        end = min(start + batch_rows, len(users))
        full_scores = user_scores[users[start:end]] @ label_matrix.T
        output[start:end] = np.take_along_axis(full_scores, candidates[start:end], axis=1)
    return output


def _evaluate_score(
    dataset: str,
    name: str,
    scores: np.ndarray,
    users: np.ndarray,
    positives: np.ndarray,
    candidates: np.ndarray,
    *,
    key_seed: int,
    k: int,
    batch_rows: int,
    randomized_repetitions: int,
    randomized_seed_start: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    for policy in (
        TiePolicy.STABLE_POSITIVE_FIRST,
        TiePolicy.ARCHIVED_FLOAT32,
        TiePolicy.HARDENED_UINT64,
    ):
        result = evaluate_policy_batched(
            candidates,
            scores,
            users,
            positives,
            policy=policy,
            key_seed=key_seed,
            k=k,
            batch_rows=batch_rows,
        )
        metric_rows.append(
            {
                "dataset": dataset,
                "score": name,
                "policy": policy.value,
                "k": k,
                **result.to_dict(),
            }
        )

    analytic = analytic_single_positive_expectation(
        scores, candidates, positives, k=k, batch_rows=batch_rows
    )
    metric_rows.append(
        {
            "dataset": dataset,
            "score": name,
            "policy": "analytic_uniform_tie_expectation",
            "k": k,
            **analytic.to_dict(),
        }
    )

    repetitions, random_summary = repeated_randomized_expectation(
        candidates,
        scores,
        users,
        positives,
        repetitions=randomized_repetitions,
        seed_start=randomized_seed_start,
        k=k,
        batch_rows=batch_rows,
    )
    repetitions.insert(0, "score", name)
    repetitions.to_csv(output_dir / f"randomized_{name}.csv", index=False)

    diagnostics = tie_diagnostics(candidates, scores, positives, k=k, batch_rows=batch_rows).to_dict()
    diagnostics.update(
        policy_difference_diagnostics(
            candidates,
            scores,
            users,
            positives,
            left_policy=TiePolicy.STABLE_POSITIVE_FIRST,
            right_policy=TiePolicy.HARDENED_UINT64,
            key_seed=key_seed,
            k=k,
            batch_rows=batch_rows,
        )
    )
    diagnostics["score"] = name
    diagnostics["randomized_summary"] = random_summary
    return metric_rows, diagnostics


def run(config: dict[str, Any]) -> dict[str, Any]:
    start_time = time.time()
    artifact = config["artifact"]
    policy = config["policies"]
    settings = config["movielens"]
    output_dir = repo_path(config, artifact["output_root"]) / "movielens"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        runtime_manifest(config["_repo_root"], config["_config_path"]),
        output_dir / "runtime_manifest.json",
    )
    write_resolved_config(config, output_dir / "resolved_config.json")

    data = load_and_preprocess(config)
    score_sources = compute_user_scores(data["train"], data["label_matrix"], data["n_users"])
    users, positives, candidates = sample_candidates_paper_compatible(
        data["train"],
        data["test"],
        data["n_items"],
        num_negatives=int(artifact["num_negatives"]),
        seed=int(settings["sampling"]["seed"]),
    )
    np.savez_compressed(
        output_dir / "frozen_candidate_rows.npz",
        user_indices=users,
        positive_items=positives,
        candidates=candidates,
    )

    all_metric_rows: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    for name in ("raw_count", "centered_count"):
        scores = score_candidate_rows(
            score_sources[name], data["label_matrix"], users, candidates
        )
        np.save(output_dir / f"candidate_scores_{name}.npy", scores)
        metric_rows, diagnostics = _evaluate_score(
            "MovieLens",
            name,
            scores,
            users,
            positives,
            candidates,
            key_seed=int(policy["movielens_key_seed"]),
            k=int(artifact["cutoff"]),
            batch_rows=int(artifact["batch_rows"]),
            randomized_repetitions=int(policy["randomized_repetitions"]),
            randomized_seed_start=int(policy["randomized_seed_start"]),
            output_dir=output_dir,
        )
        all_metric_rows.extend(metric_rows)
        all_diagnostics.append(diagnostics)

    popularity_scores = score_sources["popularity"][candidates]
    np.save(output_dir / "candidate_scores_popularity.npy", popularity_scores)
    metric_rows, diagnostics = _evaluate_score(
        "MovieLens",
        "popularity",
        popularity_scores,
        users,
        positives,
        candidates,
        key_seed=int(policy["movielens_key_seed"]),
        k=int(artifact["cutoff"]),
        batch_rows=int(artifact["batch_rows"]),
        randomized_repetitions=int(policy["randomized_repetitions"]),
        randomized_seed_start=int(policy["randomized_seed_start"]),
        output_dir=output_dir,
    )
    all_metric_rows.extend(metric_rows)
    all_diagnostics.append(diagnostics)

    metrics = pd.DataFrame(all_metric_rows)
    metrics.to_csv(output_dir / "metrics_by_policy.csv", index=False)
    with (output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(all_diagnostics, handle, indent=2)
        handle.write("\n")

    summary = {
        "dataset": {
            "n_users": data["n_users"],
            "n_items": data["n_items"],
            "n_train": len(data["train"]),
            "n_validation": len(data["validation"]),
            "n_test": len(data["test"]),
            "labels": data["label_names"],
        },
        "runtime_seconds": time.time() - start_time,
        "outputs": {
            "metrics": "metrics_by_policy.csv",
            "diagnostics": "diagnostics.json",
            "frozen_rows": "frozen_candidate_rows.npz",
            "resolved_config": "resolved_config.json",
            "runtime_manifest": "runtime_manifest.json",
            "integrity_manifest": "run_integrity_manifest.json",
        },
    }
    write_json(summary, output_dir / "run_summary.json")
    input_paths = [
        repo_path(config, "data/raw/movielens/download_manifest.json"),
        Path(config["_config_path"]),
    ]
    write_run_integrity_manifest(
        output_dir, config["_repo_root"], input_paths=input_paths
    )
    return summary
