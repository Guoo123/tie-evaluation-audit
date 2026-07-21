from __future__ import annotations

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
from .provenance import file_record, runtime_manifest, write_json
from .ranking import TiePolicy


def _score_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("candidate_scores_*.npy"))


def replay_dataset(
    config: dict[str, Any],
    dataset: str,
    *,
    include_randomized: bool = False,
    include_diagnostics: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Replay evaluator policies from a completed run's frozen rows and scores.

    This function does not download, preprocess, fit, or resample. It is the prospective
    row-level replay path that was unavailable for the historical paper run.
    """
    dataset = dataset.lower()
    if dataset not in {"bpc", "movielens"}:
        raise ValueError("dataset must be 'bpc' or 'movielens'")
    root = repo_path(config, config["artifact"]["output_root"]) / dataset
    rows_path = root / "frozen_candidate_rows.npz"
    if not rows_path.exists():
        raise FileNotFoundError(f"frozen rows not found: {rows_path}")
    score_paths = _score_files(root)
    if not score_paths:
        raise FileNotFoundError(
            f"no candidate_scores_*.npy files found under {root}; rerun with score retention enabled"
        )

    frozen = np.load(rows_path)
    users = frozen["user_indices"].astype(np.int64, copy=False)
    positives = frozen["positive_items"].astype(np.int64, copy=False)
    candidates = frozen["candidates"].astype(np.int64, copy=False)
    if candidates.ndim != 2 or len(users) != len(candidates) or len(positives) != len(candidates):
        raise ValueError("frozen candidate archive has inconsistent shapes")

    key_seed = int(
        config["policies"]["bpc_key_seed" if dataset == "bpc" else "movielens_key_seed"]
    )
    k = int(config["artifact"]["cutoff"])
    batch_rows = int(config["artifact"].get("batch_rows", 100_000))
    output_dir = output_dir or (root / "replay")
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    input_files = [rows_path]
    for score_path in score_paths:
        score_name = score_path.stem.removeprefix("candidate_scores_")
        scores = np.load(score_path, mmap_mode="r")
        if scores.shape != candidates.shape:
            raise ValueError(
                f"shape mismatch for {score_path.name}: {scores.shape} != {candidates.shape}"
            )
        input_files.append(score_path)
        for policy in (
            TiePolicy.STABLE_POSITIVE_FIRST,
            TiePolicy.ARCHIVED_FLOAT32,
            TiePolicy.HARDENED_UINT64,
        ):
            metrics = evaluate_policy_batched(
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
                    "dataset": "BPC" if dataset == "bpc" else "MovieLens",
                    "score": score_name,
                    "policy": policy.value,
                    "k": k,
                    **metrics.to_dict(),
                }
            )
        analytic = analytic_single_positive_expectation(
            scores,
            candidates,
            positives,
            k=k,
            batch_rows=batch_rows,
        )
        metric_rows.append(
            {
                "dataset": "BPC" if dataset == "bpc" else "MovieLens",
                "score": score_name,
                "policy": "analytic_uniform_tie_expectation",
                "k": k,
                **analytic.to_dict(),
            }
        )
        randomized_summary: dict[str, float] | None = None
        if include_randomized:
            frame, randomized_summary = repeated_randomized_expectation(
                candidates,
                scores,
                users,
                positives,
                repetitions=int(config["policies"]["randomized_repetitions"]),
                seed_start=int(config["policies"]["randomized_seed_start"]),
                k=k,
                batch_rows=batch_rows,
            )
            frame.insert(0, "score", score_name)
            frame.to_csv(output_dir / f"randomized_{score_name}.csv", index=False)

        if include_diagnostics:
            diagnostics = tie_diagnostics(
                candidates,
                scores,
                positives,
                k=k,
                batch_rows=batch_rows,
            ).to_dict()
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
            diagnostics["score"] = score_name
            diagnostics["randomized_summary"] = randomized_summary
            diagnostic_rows.append(diagnostics)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_path = output_dir / "metrics_by_policy.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    if include_diagnostics:
        write_json(diagnostic_rows, output_dir / "diagnostics.json")

    original_metrics_path = root / "metrics_by_policy.csv"
    comparison: dict[str, Any] = {"status": "not-available"}
    if original_metrics_path.exists():
        original = pd.read_csv(original_metrics_path)
        keys = ["dataset", "score", "policy", "k"]
        replayed = metrics_frame.merge(
            original,
            on=keys,
            how="inner",
            suffixes=("_replay", "_original"),
        )
        fields = ["hit_at_k", "ndcg_at_k", "mrr"]
        maximum = 0.0
        for field in fields:
            if f"{field}_replay" in replayed:
                delta = np.abs(
                    replayed[f"{field}_replay"].to_numpy(dtype=float)
                    - replayed[f"{field}_original"].to_numpy(dtype=float)
                )
                if len(delta):
                    maximum = max(maximum, float(delta.max()))
        expected_matches = len(score_paths) * 4
        comparison = {
            "status": (
                "pass"
                if maximum <= 1e-12 and len(replayed) == expected_matches
                else "mismatch"
            ),
            "matched_rows": int(len(replayed)),
            "expected_matched_rows": expected_matches,
            "maximum_absolute_metric_difference": maximum,
        }
        if comparison["status"] != "pass":
            raise AssertionError(
                f"frozen-row replay differs from recorded metrics (max abs diff={maximum})"
            )

    manifest = {
        "dataset": dataset,
        "runtime": runtime_manifest(config["_repo_root"], config["_config_path"]),
        "inputs": [
            file_record(path, root=Path(config["_repo_root"])) for path in input_files
        ],
        "include_randomized": include_randomized,
        "include_diagnostics": include_diagnostics,
        "recorded_metric_comparison": comparison,
        "outputs": [
            file_record(path, root=Path(config["_repo_root"]))
            for path in sorted(output_dir.glob("*"))
            if path.is_file() and path.name != "replay_manifest.json"
        ],
    }
    write_json(manifest, output_dir / "replay_manifest.json")
    return manifest
