from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .bpc_models import (
    GroupControlResult,
    build_item_groups,
    build_score_matrices,
    fit_group_control_oof,
    fit_propensities,
    score_candidate_rows,
)
from .bpc_preprocess import preprocess, stage_raw_files
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
from .sampling import history_from_frames, sample_bpc_paper_compatible, sample_unique_portable

LOGGER = logging.getLogger(__name__)


def _load_processed(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = repo_path(config, "data/processed/bpc")
    required = [
        root / "train.parquet",
        root / "validation.parquet",
        root / "test.parquet",
        root / "items.parquet",
        root / "dataset_summary.json",
    ]
    if not all(path.exists() for path in required):
        preprocess(config)
    train = pd.read_parquet(root / "train.parquet")
    validation = pd.read_parquet(root / "validation.parquet")
    test = pd.read_parquet(root / "test.parquet")
    items = pd.read_parquet(root / "items.parquet")
    with (root / "dataset_summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return train, validation, test, items, summary


def _load_or_build_groups(
    items: pd.DataFrame,
    settings: dict[str, Any],
    output_dir: Path,
    *,
    resume: bool,
) -> sparse.csr_matrix:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "item_group_matrix.npz"
    mapping_path = output_dir / "group_mapping.json"
    if resume and matrix_path.exists() and mapping_path.exists():
        return sparse.load_npz(matrix_path)
    mapping, matrix = build_item_groups(items, settings)
    sparse.save_npz(matrix_path, matrix)
    write_json(mapping, mapping_path)
    return matrix


def _load_or_build_scores(
    config: dict[str, Any],
    train: pd.DataFrame,
    items: pd.DataFrame,
    n_users: int,
    output_dir: Path,
    *,
    include_low_tie: bool,
    resume: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    score_dir = output_dir / "score_artifacts"
    score_dir.mkdir(parents=True, exist_ok=True)
    required_names = ["raw_count", "centered_count", "item_residual_only"]
    if include_low_tie:
        required_names.append("residualized_group_control")
    if resume and all(
        (score_dir / f"user_scores_{name}.npy").exists()
        and (score_dir / f"item_representation_{name}.npy").exists()
        for name in required_names
    ):
        return (
            {name: np.load(score_dir / f"user_scores_{name}.npy") for name in required_names},
            {name: np.load(score_dir / f"item_representation_{name}.npy") for name in required_names},
        )

    bpc = config["bpc"]
    label_names = list(bpc["labels"]["names"])
    propensity = fit_propensities(
        items,
        label_names,
        bpc["propensity"],
        output_dir / "propensity",
        resume=resume,
    )
    item_groups = _load_or_build_groups(
        items,
        bpc["group_control"],
        output_dir / "group_control",
        resume=resume,
    )
    group_control: GroupControlResult | None = None
    if include_low_tie:
        group_control = fit_group_control_oof(
            train,
            item_groups,
            n_users,
            bpc["group_control"],
            output_dir / "group_control",
            resume=resume,
        )
    scores, representations = build_score_matrices(
        train,
        items,
        label_names,
        propensity.probabilities,
        group_control,
        trim_above=float(bpc["propensity"]["trim_above"]),
        n_users=n_users,
    )
    for name in required_names:
        np.save(score_dir / f"user_scores_{name}.npy", scores[name])
        np.save(score_dir / f"item_representation_{name}.npy", representations[name])
    return ({name: scores[name] for name in required_names}, {name: representations[name] for name in required_names})


def _evaluate_score(
    name: str,
    candidate_scores: np.ndarray,
    users: np.ndarray,
    positives: np.ndarray,
    candidates: np.ndarray,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    artifact = config["artifact"]
    policies = config["policies"]
    key_seed = int(policies["bpc_key_seed"])
    k = int(artifact["cutoff"])
    metric_rows: list[dict[str, Any]] = []
    for policy in (
        TiePolicy.STABLE_POSITIVE_FIRST,
        TiePolicy.ARCHIVED_FLOAT32,
        TiePolicy.HARDENED_UINT64,
    ):
        result = evaluate_policy_batched(
            candidates,
            candidate_scores,
            users,
            positives,
            policy=policy,
            key_seed=key_seed,
            k=k,
            batch_rows=int(artifact["batch_rows"]),
        )
        metric_rows.append(
            {
                "dataset": "BPC",
                "score": name,
                "policy": policy.value,
                "k": k,
                **result.to_dict(),
            }
        )
    analytic = analytic_single_positive_expectation(
        candidate_scores,
        candidates,
        positives,
        k=k,
        batch_rows=int(artifact["batch_rows"]),
    )
    metric_rows.append(
        {
            "dataset": "BPC",
            "score": name,
            "policy": "analytic_uniform_tie_expectation",
            "k": k,
            **analytic.to_dict(),
        }
    )

    repetitions = int(policies.get("randomized_repetitions", 0))
    random_summary: dict[str, float] | None = None
    if repetitions > 0:
        random_frame, random_summary = repeated_randomized_expectation(
            candidates,
            candidate_scores,
            users,
            positives,
            repetitions=repetitions,
            seed_start=int(policies["randomized_seed_start"]),
            k=k,
            batch_rows=int(artifact["batch_rows"]),
        )
        random_frame.insert(0, "score", name)
        random_frame.to_csv(output_dir / f"randomized_{name}.csv", index=False)

    diagnostics = tie_diagnostics(
        candidates,
        candidate_scores,
        positives,
        k=k,
        batch_rows=int(artifact["batch_rows"]),
    ).to_dict()
    diagnostics.update(
        policy_difference_diagnostics(
            candidates,
            candidate_scores,
            users,
            positives,
            left_policy=TiePolicy.STABLE_POSITIVE_FIRST,
            right_policy=TiePolicy.HARDENED_UINT64,
            key_seed=key_seed,
            k=k,
            batch_rows=int(artifact["batch_rows"]),
        )
    )
    diagnostics["score"] = name
    diagnostics["randomized_summary"] = random_summary
    return metric_rows, diagnostics


def _finalize_summary(
    config: dict[str, Any], output_dir: Path, summary: dict[str, Any]
) -> dict[str, Any]:
    write_json(summary, output_dir / "run_summary.json")
    input_paths = [
        Path(config["_config_path"]),
        repo_path(config, "data/raw/bpc/download_manifest.json"),
        repo_path(config, "data/interim/bpc/stage_manifest.json"),
        repo_path(config, "data/processed/bpc/dataset_summary.json"),
    ]
    write_run_integrity_manifest(
        output_dir, config["_repo_root"], input_paths=input_paths
    )
    return summary


def run(
    config: dict[str, Any],
    *,
    stage: str = "all",
    max_rows: int | None = None,
    include_low_tie: bool = True,
    resume: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Run BPC regeneration with resumable stages.

    Stages are ``stage_raw``, ``preprocess``, ``model``, ``evaluate``, and ``all``.
    ``max_rows`` is a reviewer smoke-test option; paper-scale numbers require all rows.
    """
    allowed = {"stage_raw", "preprocess", "model", "evaluate", "all"}
    if stage not in allowed:
        raise ValueError(f"stage must be one of {sorted(allowed)}")
    start = time.time()
    output_dir = repo_path(config, config["artifact"]["output_root"]) / "bpc"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        runtime_manifest(config["_repo_root"], config["_config_path"]),
        output_dir / "runtime_manifest.json",
    )
    write_resolved_config(config, output_dir / "resolved_config.json")

    if stage in {"stage_raw", "all"}:
        stage_raw_files(config, force=force)
        if stage == "stage_raw":
            return _finalize_summary(
                config,
                output_dir,
                {"stage": stage, "output": "data/interim/bpc"},
            )
    if stage in {"preprocess", "all"}:
        preprocess(config, force=force)
        if stage == "preprocess":
            return _finalize_summary(
                config,
                output_dir,
                {"stage": stage, "output": "data/processed/bpc"},
            )

    train, validation, test, items, dataset_summary = _load_processed(config)
    n_users = int(dataset_summary["n_users"])
    n_items = int(dataset_summary["n_items"])
    effective_resume = resume and not force
    scores, representations = _load_or_build_scores(
        config,
        train,
        items,
        n_users,
        output_dir,
        include_low_tie=include_low_tie,
        resume=effective_resume,
    )
    if stage == "model":
        return _finalize_summary(
            config,
            output_dir,
            {"stage": stage, "scores": sorted(scores), "output": "score_artifacts"},
        )

    evaluation_test = test.iloc[:max_rows].copy() if max_rows is not None else test
    history = history_from_frames([train, validation], n_users)
    sampling = config["bpc"]["sampling"]
    if sampling["mode"] == "paper_compatible_rejection":
        candidate_rows = sample_bpc_paper_compatible(
            evaluation_test,
            history,
            n_items=n_items,
            num_negatives=int(config["artifact"]["num_negatives"]),
            seed=int(sampling["seed"]),
        )
    else:
        candidate_rows = sample_unique_portable(
            evaluation_test,
            history,
            n_items=n_items,
            num_negatives=int(config["artifact"]["num_negatives"]),
            seed=int(sampling["seed"]),
        )
    if not np.all(candidate_rows.candidate_counts == int(config["artifact"]["candidate_count"])):
        raise RuntimeError("one or more BPC rows did not receive the requested number of negatives")

    if bool(config["artifact"].get("save_candidate_rows", True)):
        np.savez_compressed(
            output_dir / "frozen_candidate_rows.npz",
            user_indices=candidate_rows.user_indices,
            positive_items=candidate_rows.positive_items,
            candidates=candidate_rows.candidates,
        )

    all_metrics: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    names = ["raw_count", "centered_count"]
    if include_low_tie:
        names.append("residualized_group_control")
    # Preserve the simpler item-only residual as an explicitly named diagnostic;
    # it is not the source of the paper's 0.1689/0.1685 row.
    names.append("item_residual_only")
    for name in names:
        candidate_scores = score_candidate_rows(
            scores[name],
            representations[name],
            candidate_rows.user_indices,
            candidate_rows.candidates,
            batch_rows=int(config["artifact"]["batch_rows"]),
        )
        metric_rows, diagnostics = _evaluate_score(
            name,
            candidate_scores,
            candidate_rows.user_indices,
            candidate_rows.positive_items,
            candidate_rows.candidates,
            config,
            output_dir,
        )
        all_metrics.extend(metric_rows)
        all_diagnostics.append(diagnostics)
        if bool(config["artifact"].get("save_candidate_scores", False)):
            np.save(output_dir / f"candidate_scores_{name}.npy", candidate_scores)

    metric_frame = pd.DataFrame(all_metrics)
    metric_frame.to_csv(output_dir / "metrics_by_policy.csv", index=False)
    write_json(all_diagnostics, output_dir / "diagnostics.json")
    summary = {
        "dataset": dataset_summary,
        "evaluated_rows": len(evaluation_test),
        "paper_scale": max_rows is None,
        "score_families": names,
        "runtime_seconds": time.time() - start,
        "outputs": {
            "metrics": "metrics_by_policy.csv",
            "diagnostics": "diagnostics.json",
            "candidate_rows": "frozen_candidate_rows.npz"
            if bool(config["artifact"].get("save_candidate_rows", True))
            else None,
            "candidate_scores_pattern": "candidate_scores_*.npy",
            "resolved_config": "resolved_config.json",
            "runtime_manifest": "runtime_manifest.json",
            "integrity_manifest": "run_integrity_manifest.json",
        },
    }
    return _finalize_summary(config, output_dir, summary)
