from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .labels import BEAUTY_LABELS, apply_target_mask
from .provenance import write_json
from .text import normalize_text, price_bucket

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PropensityResult:
    probabilities: dict[str, np.ndarray]
    metrics: pd.DataFrame


@dataclass(frozen=True)
class GroupControlResult:
    fold_ids: np.ndarray
    user_oof: np.ndarray
    group_oof: np.ndarray


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if item is not None]
            except Exception:
                pass
        return [value]
    return [str(value)]


def build_item_groups(items: pd.DataFrame, settings: dict[str, Any]) -> tuple[dict[str, int], sparse.csr_matrix]:
    raw_groups: list[list[str]] = []
    for row in items.itertuples(index=False):
        groups: list[str] = []
        if settings["include_store"] and getattr(row, "store", ""):
            store = normalize_text(str(getattr(row, "store")))
            if store:
                groups.append(f"store::{store}")
        if settings["include_category"]:
            main = normalize_text(str(getattr(row, "main_category", "") or ""))
            if main:
                groups.append(f"main_category::{main}")
            for category in _safe_list(getattr(row, "categories", None))[:3]:
                category = normalize_text(category)
                if category:
                    groups.append(f"category::{category}")
        if settings["include_price_bucket"]:
            groups.append(price_bucket(getattr(row, "price_num", None)))
        raw_groups.append(sorted(set(groups)))

    support: dict[str, int] = {}
    for groups in raw_groups:
        for group in groups:
            support[group] = support.get(group, 0) + 1
    names = sorted(
        group for group, count in support.items() if count >= int(settings["min_item_group_support"])
    )
    group_to_idx = {name: index for index, name in enumerate(names)}
    rows: list[int] = []
    columns: list[int] = []
    for item_index, groups in enumerate(raw_groups):
        for group in groups:
            if group in group_to_idx:
                rows.append(item_index)
                columns.append(group_to_idx[group])
    values = np.ones(len(rows), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (values, (np.asarray(rows), np.asarray(columns))),
        shape=(len(items), len(group_to_idx)),
    )
    return group_to_idx, matrix


def fit_propensities(
    items: pd.DataFrame,
    label_names: list[str],
    settings: dict[str, Any],
    output_dir: Path,
    *,
    resume: bool = True,
) -> PropensityResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, float | str]] = []
    for name in label_names:
        probability_path = output_dir / f"oof_prob_{name}.npy"
        metrics_path = output_dir / f"metrics_{name}.json"
        if resume and probability_path.exists() and metrics_path.exists():
            probabilities[name] = np.load(probability_path)
            with metrics_path.open("r", encoding="utf-8") as handle:
                metric_rows.append(json.load(handle))
            continue

        label_column = f"label__{name}"
        frame = items[["item_id", "item_text", label_column]].dropna(subset=[label_column]).copy()
        y = frame[label_column].to_numpy(dtype=np.int64)
        positives = int(y.sum())
        negatives = int((1 - y).sum())
        folds = min(int(settings["folds"]), positives, negatives)
        if folds < 2:
            raise ValueError(f"label {name} does not have enough examples for cross-fitting")
        masked = apply_target_mask(frame["item_text"], BEAUTY_LABELS[name])
        oof = np.zeros(len(frame), dtype=np.float32)
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=int(settings["random_seed"]),
        )
        for fold, (training_indices, validation_indices) in enumerate(splitter.split(masked, y)):
            vectorizer = TfidfVectorizer(
                max_features=int(settings["max_features"]),
                min_df=int(settings["min_df"]),
                ngram_range=(1, 2),
            )
            x_train = vectorizer.fit_transform(masked.iloc[training_indices])
            x_validation = vectorizer.transform(masked.iloc[validation_indices])
            classifier = LogisticRegression(
                C=float(settings["C"]),
                max_iter=int(settings["max_iter"]),
                solver="liblinear",
                class_weight=str(settings["class_weight"]),
                random_state=int(settings["random_seed"]) + fold,
            )
            classifier.fit(x_train, y[training_indices])
            oof[validation_indices] = classifier.predict_proba(x_validation)[:, 1]
            LOGGER.info("BPC propensity %s fold %d/%d", name, fold + 1, folds)

        full = np.full(len(items), np.nan, dtype=np.float32)
        full[frame.index.to_numpy()] = oof
        metrics = {
            "label": name,
            "positive_rate": float(y.mean()),
            "log_loss": float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y, oof)),
            "auroc": float(roc_auc_score(y, oof)) if len(np.unique(y)) > 1 else float("nan"),
            "num_items": int(len(frame)),
            "num_positive_items": positives,
        }
        np.save(probability_path, full)
        write_json(metrics, metrics_path)
        probabilities[name] = full
        metric_rows.append(metrics)

        # A full-data model is not needed for the paper's score, but retaining one
        # makes the preprocessing artifact independently inspectable.
        vectorizer = TfidfVectorizer(
            max_features=int(settings["max_features"]),
            min_df=int(settings["min_df"]),
            ngram_range=(1, 2),
        )
        x_full = vectorizer.fit_transform(masked)
        classifier = LogisticRegression(
            C=float(settings["C"]),
            max_iter=int(settings["max_iter"]),
            solver="liblinear",
            class_weight=str(settings["class_weight"]),
            random_state=int(settings["random_seed"]),
        )
        classifier.fit(x_full, y)
        joblib.dump(
            {"vectorizer": vectorizer, "classifier": classifier, "label_name": name},
            output_dir / f"model_{name}.joblib",
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(output_dir / "propensity_metrics.csv", index=False)
    return PropensityResult(probabilities=probabilities, metrics=metrics_frame)


def assign_folds_within_user(train: pd.DataFrame, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    temporary = pd.DataFrame(
        {
            "user_idx": train["user_idx"].to_numpy(dtype=np.int64),
            "random_value": rng.random(len(train)),
            "original_index": np.arange(len(train), dtype=np.int64),
        }
    )
    temporary = temporary.sort_values(["user_idx", "random_value"])
    temporary["fold_id"] = temporary.groupby("user_idx").cumcount() % folds
    temporary = temporary.sort_values("original_index")
    return temporary["fold_id"].to_numpy(dtype=np.int32)


def _fit_user_group(
    train: pd.DataFrame,
    item_groups: sparse.csr_matrix,
    n_users: int,
    *,
    smoothing: float,
    minimum_count: int,
) -> tuple[float, np.ndarray, dict[tuple[int, int], float]]:
    y = train["y"].to_numpy(dtype=np.float32)
    users = train["user_idx"].to_numpy(dtype=np.int64)
    items = train["item_idx"].to_numpy(dtype=np.int64)
    global_mean = float(y.mean()) if len(y) else 0.0
    user_sum = np.bincount(users, weights=y, minlength=n_users).astype(np.float32)
    user_count = np.bincount(users, minlength=n_users).astype(np.float32)
    user_bias = (user_sum + smoothing * global_mean) / (user_count + smoothing)
    deviation = y - user_bias[users]

    pair_sum: dict[tuple[int, int], float] = {}
    pair_count: dict[tuple[int, int], int] = {}
    indptr = item_groups.indptr
    group_indices = item_groups.indices
    for row, (user, item) in enumerate(zip(users, items)):
        start, end = indptr[item], indptr[item + 1]
        value = float(deviation[row])
        for group in group_indices[start:end]:
            key = (int(user), int(group))
            pair_sum[key] = pair_sum.get(key, 0.0) + value
            pair_count[key] = pair_count.get(key, 0) + 1
    coefficients = {
        key: total / (pair_count[key] + smoothing)
        for key, total in pair_sum.items()
        if pair_count[key] >= minimum_count
    }
    return global_mean, user_bias.astype(np.float32), coefficients


def _predict_user_group(
    frame: pd.DataFrame,
    item_groups: sparse.csr_matrix,
    global_mean: float,
    user_bias: np.ndarray,
    coefficients: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    users = frame["user_idx"].to_numpy(dtype=np.int64)
    items = frame["item_idx"].to_numpy(dtype=np.int64)
    user_prediction = np.full(len(frame), global_mean, dtype=np.float32)
    valid = (users >= 0) & (users < len(user_bias))
    user_prediction[valid] = user_bias[users[valid]]
    group_prediction = np.zeros(len(frame), dtype=np.float32)
    indptr = item_groups.indptr
    group_indices = item_groups.indices
    for row, (user, item) in enumerate(zip(users, items)):
        for group in group_indices[indptr[item] : indptr[item + 1]]:
            group_prediction[row] += coefficients.get((int(user), int(group)), 0.0)
    return user_prediction, group_prediction


def fit_group_control_oof(
    train: pd.DataFrame,
    item_groups: sparse.csr_matrix,
    n_users: int,
    settings: dict[str, Any],
    output_dir: Path,
    *,
    resume: bool = True,
) -> GroupControlResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "group_control_oof.npz"
    if resume and path.exists():
        loaded = np.load(path)
        return GroupControlResult(loaded["fold_ids"], loaded["user_oof"], loaded["group_oof"])

    folds = int(settings["folds"])
    fold_ids = assign_folds_within_user(train, folds, int(settings["random_seed"]))
    user_oof = np.full(len(train), np.nan, dtype=np.float32)
    group_oof = np.full(len(train), np.nan, dtype=np.float32)
    for fold in range(folds):
        holdout_mask = fold_ids == fold
        training_mask = ~holdout_mask
        fold_train = train.iloc[training_mask].reset_index(drop=True)
        fold_holdout = train.iloc[holdout_mask].reset_index(drop=True)
        LOGGER.info(
            "BPC group control fold %d/%d: train=%d holdout=%d",
            fold + 1,
            folds,
            len(fold_train),
            len(fold_holdout),
        )
        global_mean, user_bias, coefficients = _fit_user_group(
            fold_train,
            item_groups,
            n_users,
            smoothing=float(settings["smoothing"]),
            minimum_count=int(settings["min_user_group_count"]),
        )
        user_prediction, group_prediction = _predict_user_group(
            fold_holdout, item_groups, global_mean, user_bias, coefficients
        )
        holdout_indices = np.flatnonzero(holdout_mask)
        user_oof[holdout_indices] = user_prediction
        group_oof[holdout_indices] = group_prediction
    if not np.all(np.isfinite(user_oof)) or not np.all(np.isfinite(group_oof)):
        raise RuntimeError("cross-fitted group control produced non-finite predictions")
    np.savez_compressed(path, fold_ids=fold_ids, user_oof=user_oof, group_oof=group_oof)
    return GroupControlResult(fold_ids, user_oof, group_oof)


def aggregate_user_scores(
    users: np.ndarray,
    items: np.ndarray,
    item_matrix: np.ndarray,
    weights: np.ndarray,
    n_users: int,
    *,
    chunk_rows: int = 250_000,
) -> np.ndarray:
    output = np.zeros((n_users, item_matrix.shape[1]), dtype=np.float32)
    for start in range(0, len(users), chunk_rows):
        end = min(start + chunk_rows, len(users))
        contributions = weights[start:end, None].astype(np.float32) * item_matrix[items[start:end]]
        np.add.at(output, users[start:end], contributions)
    return output


def build_score_matrices(
    train: pd.DataFrame,
    items: pd.DataFrame,
    label_names: list[str],
    propensities: dict[str, np.ndarray],
    group_control: GroupControlResult | None,
    *,
    trim_above: float,
    n_users: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    label_matrix = np.stack(
        [items[f"label__{name}"].fillna(0).to_numpy(dtype=np.float32) for name in label_names],
        axis=1,
    )
    p_hat = np.stack(
        [
            np.nan_to_num(propensities[name], nan=float(label_matrix[:, index].mean()))
            for index, name in enumerate(label_names)
        ],
        axis=1,
    ).astype(np.float32)
    residual_matrix = label_matrix - p_hat
    residual_matrix = np.where(p_hat > trim_above, 0.0, residual_matrix).astype(np.float32)
    centered_matrix = (label_matrix - label_matrix.mean(axis=0, keepdims=True)).astype(np.float32)

    users = train["user_idx"].to_numpy(dtype=np.int64)
    item_indices = train["item_idx"].to_numpy(dtype=np.int64)
    y = train["y"].to_numpy(dtype=np.float32)
    scores = {
        "raw_count": aggregate_user_scores(users, item_indices, label_matrix, y, n_users),
        "centered_count": aggregate_user_scores(users, item_indices, centered_matrix, y, n_users),
        "item_residual_only": aggregate_user_scores(users, item_indices, residual_matrix, y, n_users),
    }
    representations = {
        "raw_count": label_matrix,
        "centered_count": centered_matrix,
        "item_residual_only": residual_matrix,
    }
    if group_control is not None:
        residual_outcome = y - group_control.user_oof - group_control.group_oof
        scores["residualized_group_control"] = aggregate_user_scores(
            users, item_indices, residual_matrix, residual_outcome, n_users
        )
        representations["residualized_group_control"] = residual_matrix
    return scores, representations


def score_candidate_rows(
    user_scores: np.ndarray,
    item_representation: np.ndarray,
    users: np.ndarray,
    candidates: np.ndarray,
    *,
    batch_rows: int = 100_000,
) -> np.ndarray:
    output = np.empty(candidates.shape, dtype=np.float32)
    for start in range(0, len(users), batch_rows):
        end = min(start + batch_rows, len(users))
        output[start:end] = np.einsum(
            "ijk,ik->ij",
            item_representation[candidates[start:end]],
            user_scores[users[start:end]],
        )
    return output
