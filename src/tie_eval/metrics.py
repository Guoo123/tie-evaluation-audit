from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .ranking import TiePolicy, rank_candidates


@dataclass(frozen=True)
class AggregateMetrics:
    n_rows: int
    hit_at_k: float
    ndcg_at_k: float
    mrr: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def positive_ranks(ranked_candidates: np.ndarray, positive_items: np.ndarray) -> np.ndarray:
    ranked = np.asarray(ranked_candidates)
    positives = np.asarray(positive_items, dtype=np.int64)
    if ranked.ndim != 2:
        raise ValueError("ranked_candidates must be two-dimensional")
    if positives.ndim != 1 or len(positives) != len(ranked):
        raise ValueError("positive_items must contain one item identity per row")
    labels = ranked == positives[:, None]
    occurrences = labels.sum(axis=1)
    if not np.all(occurrences == 1):
        missing = int((occurrences == 0).sum())
        duplicated = int((occurrences > 1).sum())
        raise ValueError(
            "each candidate row must contain the positive exactly once "
            f"(missing={missing}, duplicated={duplicated})"
        )
    return np.argmax(labels, axis=1).astype(np.int64) + 1


def aggregate_from_ranks(ranks: np.ndarray, k: int) -> AggregateMetrics:
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or len(ranks) == 0:
        raise ValueError("ranks must be a non-empty one-dimensional array")
    if k < 1:
        raise ValueError("k must be positive")
    if np.any(ranks < 1):
        raise ValueError("ranks are one-indexed and must be at least one")
    in_topk = ranks <= k
    ndcg = np.where(in_topk, 1.0 / np.log2(ranks + 1.0), 0.0)
    return AggregateMetrics(
        n_rows=int(len(ranks)),
        hit_at_k=float(np.mean(in_topk)),
        ndcg_at_k=float(np.mean(ndcg)),
        mrr=float(np.mean(1.0 / ranks)),
    )


def _validate_evaluation_inputs(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    user_indices: np.ndarray,
    positive_items: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(candidate_matrix)
    scores = np.asarray(score_matrix)
    users = np.asarray(user_indices, dtype=np.int64)
    positives = np.asarray(positive_items, dtype=np.int64)
    if candidates.ndim != 2 or scores.ndim != 2 or candidates.shape != scores.shape:
        raise ValueError("candidate_matrix and score_matrix must have the same 2D shape")
    if len(candidates) == 0:
        raise ValueError("at least one candidate row is required")
    if users.ndim != 1 or len(users) != len(candidates):
        raise ValueError("user_indices must contain one identity per row")
    if positives.ndim != 1 or len(positives) != len(candidates):
        raise ValueError("positive_items must contain one identity per row")
    if not 1 <= k <= candidates.shape[1]:
        raise ValueError("k must be between one and the candidate-row width")
    if not np.all(np.isfinite(scores)):
        raise ValueError("primary scores contain NaN or infinity")
    return candidates, scores, users, positives


def evaluate_policy(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    user_indices: np.ndarray,
    positive_items: np.ndarray,
    *,
    policy: str | TiePolicy,
    key_seed: int,
    k: int = 10,
    random_seed: int | None = None,
    return_ranks: bool = False,
) -> AggregateMetrics | tuple[AggregateMetrics, np.ndarray, np.ndarray]:
    candidates, scores, users, positives = _validate_evaluation_inputs(
        candidate_matrix, score_matrix, user_indices, positive_items, k=k
    )
    ranked = rank_candidates(
        candidates,
        scores,
        users,
        policy,
        seed=key_seed,
        random_seed=random_seed,
    )
    ranks = positive_ranks(ranked, positives)
    metrics = aggregate_from_ranks(ranks, k)
    if return_ranks:
        return metrics, ranks, ranked
    return metrics


def evaluate_policy_batched(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    user_indices: np.ndarray,
    positive_items: np.ndarray,
    *,
    policy: str | TiePolicy,
    key_seed: int,
    k: int = 10,
    random_seed: int | None = None,
    batch_rows: int = 100_000,
) -> AggregateMetrics:
    """Evaluate a tie policy without materializing a full ranked matrix.

    The paper-scale BPC run has more than two million 31-item rows. Ranking those rows
    in one operation creates several large temporary matrices. This implementation
    processes rows in bounded-memory batches and accumulates only metric sums.

    For ``randomized`` ordering, one RNG is advanced across batches, so the result is
    independent of the chosen batch size and matches a single row-major key draw.
    """
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    candidates, scores, users, positives = _validate_evaluation_inputs(
        candidate_matrix, score_matrix, user_indices, positive_items, k=k
    )
    selected_policy = TiePolicy(policy)
    rng = None
    if selected_policy is TiePolicy.RANDOMIZED:
        rng = np.random.default_rng(key_seed if random_seed is None else random_seed)

    hit_sum = 0.0
    ndcg_sum = 0.0
    reciprocal_sum = 0.0
    n_rows = len(candidates)
    for start in range(0, n_rows, batch_rows):
        end = min(start + batch_rows, n_rows)
        batch_candidates = candidates[start:end]
        batch_scores = scores[start:end]
        batch_users = users[start:end]
        if selected_policy is TiePolicy.RANDOMIZED:
            assert rng is not None
            random_keys = rng.random(size=batch_scores.shape, dtype=np.float64)
            order = np.lexsort((batch_candidates, random_keys, -batch_scores), axis=1)
            ranked = np.take_along_axis(batch_candidates, order, axis=1)
        else:
            ranked = rank_candidates(
                batch_candidates,
                batch_scores,
                batch_users,
                selected_policy,
                seed=key_seed,
            )
        ranks = positive_ranks(ranked, positives[start:end])
        in_topk = ranks <= k
        hit_sum += float(np.count_nonzero(in_topk))
        ndcg_sum += float(np.sum(np.where(in_topk, 1.0 / np.log2(ranks + 1.0), 0.0)))
        reciprocal_sum += float(np.sum(1.0 / ranks))

    return AggregateMetrics(
        n_rows=n_rows,
        hit_at_k=hit_sum / n_rows,
        ndcg_at_k=ndcg_sum / n_rows,
        mrr=reciprocal_sum / n_rows,
    )


def analytic_single_positive_expectation(
    score_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    positive_items: np.ndarray,
    *,
    k: int = 10,
    batch_rows: int = 250_000,
) -> AggregateMetrics:
    """Expected metrics under uniform ordering inside each exact tie block.

    For each row, ``a`` candidates score strictly above the positive and ``t``
    candidates share its score; the positive rank is uniform on
    ``{a+1, ..., a+t}``. The implementation is bounded-memory.
    """
    scores = np.asarray(score_matrix)
    candidates = np.asarray(candidate_matrix)
    positives = np.asarray(positive_items)
    if scores.ndim != 2 or candidates.shape != scores.shape:
        raise ValueError("candidate_matrix and score_matrix must have the same 2D shape")
    if positives.ndim != 1 or len(positives) != len(scores):
        raise ValueError("positive_items must contain one identity per row")
    if not np.all(np.isfinite(scores)):
        raise ValueError("primary scores contain NaN or infinity")
    n_rows, width = scores.shape
    if n_rows == 0:
        raise ValueError("at least one row is required")
    if not 1 <= k <= width:
        raise ValueError("k must be between one and the row width")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")

    ranks = np.arange(1, width + 1, dtype=np.float64)
    hit_contrib = (ranks <= k).astype(np.float64)
    ndcg_contrib = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    mrr_contrib = 1.0 / ranks
    hit_prefix = np.concatenate([[0.0], np.cumsum(hit_contrib)])
    ndcg_prefix = np.concatenate([[0.0], np.cumsum(ndcg_contrib)])
    mrr_prefix = np.concatenate([[0.0], np.cumsum(mrr_contrib)])

    hit_sum = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    for start in range(0, n_rows, batch_rows):
        end = min(start + batch_rows, n_rows)
        batch_candidates = candidates[start:end]
        batch_scores = scores[start:end]
        batch_positives = positives[start:end]
        matches = batch_candidates == batch_positives[:, None]
        occurrences = matches.sum(axis=1)
        if not np.all(occurrences == 1):
            raise ValueError("analytic expectation requires exactly one positive occurrence per row")
        positive_columns = np.argmax(matches, axis=1)
        positive_scores = batch_scores[np.arange(end - start), positive_columns]
        above = (batch_scores > positive_scores[:, None]).sum(axis=1).astype(np.int64)
        tie_size = (batch_scores == positive_scores[:, None]).sum(axis=1).astype(np.int64)
        lower = above
        upper = above + tie_size
        denominator = tie_size.astype(np.float64)
        hit_sum += float(np.sum((hit_prefix[upper] - hit_prefix[lower]) / denominator))
        ndcg_sum += float(np.sum((ndcg_prefix[upper] - ndcg_prefix[lower]) / denominator))
        mrr_sum += float(np.sum((mrr_prefix[upper] - mrr_prefix[lower]) / denominator))

    return AggregateMetrics(
        n_rows=n_rows,
        hit_at_k=hit_sum / n_rows,
        ndcg_at_k=ndcg_sum / n_rows,
        mrr=mrr_sum / n_rows,
    )


def repeated_randomized_expectation(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    user_indices: np.ndarray,
    positive_items: np.ndarray,
    *,
    repetitions: int,
    seed_start: int,
    k: int = 10,
    batch_rows: int = 100_000,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if repetitions < 1:
        raise ValueError("repetitions must be at least one")
    rows = []
    for repetition in range(repetitions):
        seed = seed_start + repetition
        result = evaluate_policy_batched(
            candidate_matrix,
            score_matrix,
            user_indices,
            positive_items,
            policy=TiePolicy.RANDOMIZED,
            key_seed=seed,
            random_seed=seed,
            k=k,
            batch_rows=batch_rows,
        )
        rows.append({"repetition": repetition, "seed": seed, **result.to_dict()})
    frame = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for name in ("hit_at_k", "ndcg_at_k", "mrr"):
        values = frame[name].to_numpy(dtype=np.float64)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f"{name}_min"] = float(values.min())
        summary[f"{name}_max"] = float(values.max())
    return frame, summary


def metrics_table(rows: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    preferred = [
        "dataset",
        "score",
        "policy",
        "n_rows",
        "ndcg_at_k",
        "hit_at_k",
        "mrr",
    ]
    return frame[
        [column for column in preferred if column in frame.columns]
        + [column for column in frame.columns if column not in preferred]
    ]
