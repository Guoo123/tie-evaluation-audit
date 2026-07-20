from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .metrics import positive_ranks
from .ranking import TiePolicy, rank_candidates


@dataclass(frozen=True)
class TieDiagnostics:
    n_rows: int
    any_tie_rate: float
    boundary_tie_rate: float
    positive_in_tie_rate: float
    average_max_tie_block: float
    median_max_tie_block: float
    average_positive_tie_size: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _validate_positive_occurrences(candidates: np.ndarray, positives: np.ndarray) -> None:
    occurrences = (candidates == positives[:, None]).sum(axis=1)
    if not np.all(occurrences == 1):
        missing = int((occurrences == 0).sum())
        duplicated = int((occurrences > 1).sum())
        raise ValueError(
            "each row must contain the positive exactly once "
            f"(missing={missing}, duplicated={duplicated})"
        )


def _histogram_median(histogram: np.ndarray, total: int) -> float:
    """Return NumPy's median for an integer-valued distribution histogram."""
    cumulative = np.cumsum(histogram)

    def value_at(zero_based_index: int) -> int:
        return int(np.searchsorted(cumulative, zero_based_index + 1, side="left"))

    if total % 2:
        return float(value_at(total // 2))
    left = value_at(total // 2 - 1)
    right = value_at(total // 2)
    return (left + right) / 2.0


def tie_diagnostics(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    positive_items: np.ndarray,
    *,
    k: int = 10,
    batch_rows: int = 100_000,
) -> TieDiagnostics:
    """Compute exact-score tie diagnostics in bounded-memory batches."""
    candidates = np.asarray(candidate_matrix)
    scores = np.asarray(score_matrix)
    positives = np.asarray(positive_items)
    if candidates.ndim != 2 or candidates.shape != scores.shape:
        raise ValueError("candidate and score matrices must have identical 2D shapes")
    n_rows, width = scores.shape
    if n_rows == 0:
        raise ValueError("at least one row is required")
    if positives.ndim != 1 or len(positives) != n_rows:
        raise ValueError("positive_items must contain one identity per row")
    if not 1 <= k < width:
        raise ValueError("k must be between 1 and number_of_candidates - 1")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    if not np.all(np.isfinite(scores)):
        raise ValueError("primary scores contain NaN or infinity")

    any_tie_total = 0
    boundary_total = 0
    positive_tie_total = 0
    max_block_sum = 0
    max_block_histogram = np.zeros(width + 1, dtype=np.int64)
    positive_tie_size_sum = 0

    for start in range(0, n_rows, batch_rows):
        end = min(start + batch_rows, n_rows)
        batch_scores = scores[start:end]
        batch_candidates = candidates[start:end]
        batch_positives = positives[start:end]
        _validate_positive_occurrences(batch_candidates, batch_positives)

        sorted_scores = np.sort(batch_scores, axis=1)[:, ::-1]
        adjacent_equal = sorted_scores[:, 1:] == sorted_scores[:, :-1]
        any_tie_total += int(adjacent_equal.any(axis=1).sum())

        boundary_score = sorted_scores[:, k - 1]
        count_gt = (batch_scores > boundary_score[:, None]).sum(axis=1)
        count_ge = (batch_scores >= boundary_score[:, None]).sum(axis=1)
        boundary_total += int(((count_gt < k) & (count_ge > k)).sum())

        for row_scores in sorted_scores:
            _, counts = np.unique(row_scores, return_counts=True)
            maximum = int(counts.max())
            max_block_sum += maximum
            max_block_histogram[maximum] += 1

        positive_positions = np.argmax(
            batch_candidates == batch_positives[:, None], axis=1
        )
        positive_scores = batch_scores[np.arange(end - start), positive_positions]
        tie_sizes = (batch_scores == positive_scores[:, None]).sum(axis=1).astype(np.int64)
        tied = tie_sizes > 1
        positive_tie_total += int(tied.sum())
        positive_tie_size_sum += int(tie_sizes[tied].sum())

    return TieDiagnostics(
        n_rows=n_rows,
        any_tie_rate=float(any_tie_total / n_rows),
        boundary_tie_rate=float(boundary_total / n_rows),
        positive_in_tie_rate=float(positive_tie_total / n_rows),
        average_max_tie_block=float(max_block_sum / n_rows),
        median_max_tie_block=_histogram_median(max_block_histogram, n_rows),
        average_positive_tie_size=(
            float(positive_tie_size_sum / positive_tie_total)
            if positive_tie_total
            else 0.0
        ),
    )


def _topk_set_changed(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compare top-k item sets, ignoring item order."""
    a_in_b = (left[:, :, None] == right[:, None, :]).any(axis=2).all(axis=1)
    b_in_a = (right[:, :, None] == left[:, None, :]).any(axis=2).all(axis=1)
    return ~(a_in_b & b_in_a)


def policy_difference_diagnostics(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    user_indices: np.ndarray,
    positive_items: np.ndarray,
    *,
    left_policy: str | TiePolicy,
    right_policy: str | TiePolicy,
    key_seed: int,
    k: int = 10,
    batch_rows: int = 100_000,
) -> dict[str, float | int]:
    """Compare two policies without retaining two full ranked datasets."""
    candidates = np.asarray(candidate_matrix)
    scores = np.asarray(score_matrix)
    users = np.asarray(user_indices)
    positives = np.asarray(positive_items)
    if candidates.ndim != 2 or candidates.shape != scores.shape:
        raise ValueError("candidate and score matrices must have identical 2D shapes")
    if len(users) != len(candidates) or len(positives) != len(candidates):
        raise ValueError("users and positives must contain one value per row")
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")

    n_rows = len(candidates)
    topk_changed_total = 0
    rank_changed_total = 0
    absolute_rank_change_sum = 0.0
    for start in range(0, n_rows, batch_rows):
        end = min(start + batch_rows, n_rows)
        batch_candidates = candidates[start:end]
        batch_scores = scores[start:end]
        batch_users = users[start:end]
        batch_positives = positives[start:end]
        left = rank_candidates(
            batch_candidates, batch_scores, batch_users, left_policy, seed=key_seed
        )
        right = rank_candidates(
            batch_candidates, batch_scores, batch_users, right_policy, seed=key_seed
        )
        left_ranks = positive_ranks(left, batch_positives)
        right_ranks = positive_ranks(right, batch_positives)
        topk_changed_total += int(_topk_set_changed(left[:, :k], right[:, :k]).sum())
        changed = left_ranks != right_ranks
        rank_changed_total += int(changed.sum())
        absolute_rank_change_sum += float(
            np.abs(left_ranks - right_ranks)[changed].sum()
        )

    return {
        "n_rows": int(n_rows),
        "topk_item_set_change_rate": float(topk_changed_total / n_rows),
        "positive_rank_change_rate": float(rank_changed_total / n_rows),
        "mean_absolute_positive_rank_change": (
            float(absolute_rank_change_sum / rank_changed_total)
            if rank_changed_total
            else 0.0
        ),
    }
