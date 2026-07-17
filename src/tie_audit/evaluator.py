"""Auditable ranking evaluation under score ties.

The module intentionally separates four policies:

``stable``
    Score-only sorting. Equal scores retain candidate input position. This is a
    useful diagnostic baseline, but it is unsafe when a positive is inserted at
    a privileged position.
``archived_float32``
    Exact reproduction of the manuscript-described BPC evaluator: a 64-bit
    mixer is divided by ``2**64``, rounded to ``float32``, and used in a
    two-pass sort. Keep this policy only for reproducing archived result files.
``hardened_uint64``
    The normative deterministic policy. It retains the full ``uint64`` mixer
    output and uses item identity as the final fallback if keys collide.
``randomized``
    Seeded Monte Carlo tie ordering with independent secondary keys. Scores are
    never jittered or otherwise modified.

All public evaluators operate on rectangular, row-major NumPy-compatible
matrices. Candidate and row identities must be non-negative integers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ARCHIVED_SEED = 20260316
"""Seed used by the archived calculation code."""

GOLDEN = np.uint64(0x9E3779B97F4A7C15)
MIX_1 = np.uint64(0xFF51AFD7ED558CCD)
MIX_2 = np.uint64(0xC4CEB9FE1A85EC53)
UINT64_MAX = int(np.iinfo(np.uint64).max)

POLICIES = ("stable", "archived_float32", "hardened_uint64", "randomized")
_POLICY_ALIASES = {
    "archived": "archived_float32",
    "hardened": "hardened_uint64",
}


def _score_matrix(scores: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"scores must be a 2-D matrix; got shape {matrix.shape}")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must contain at least one row and one candidate")
    if np.isnan(matrix).any():
        raise ValueError("scores must not contain NaN")
    return matrix


def _id_array(
    values: np.ndarray | Sequence[int] | Sequence[Sequence[int]],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must contain integer identities")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError(f"{name} must contain non-negative identities")
    return array.astype(np.uint64, copy=False)


def _seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    value = int(seed)
    if value < 0 or value > UINT64_MAX:
        raise ValueError("seed must be in the uint64 range")
    return value


def _row_id_array(
    row_ids: np.ndarray | Sequence[int] | None,
    n_rows: int,
) -> np.ndarray:
    if row_ids is None:
        return np.arange(n_rows, dtype=np.uint64)
    return _id_array(row_ids, (n_rows,), "row_ids")


def _item_matrix(
    item_ids: np.ndarray | Sequence[Sequence[int]],
    shape: tuple[int, int],
) -> np.ndarray:
    return _id_array(item_ids, shape, "item_ids")


def _require_unique_items(item_ids: np.ndarray) -> None:
    for row_number, row in enumerate(item_ids):
        if np.unique(row).size != row.size:
            raise ValueError(
                "hardened ordering requires unique item identities within each "
                f"row; row {row_number} contains a duplicate"
            )


def _validated_order(
    order: np.ndarray | Sequence[Sequence[int]],
    shape: tuple[int, int],
    name: str = "order",
) -> np.ndarray:
    array = np.asarray(order)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer positions")
    expected = np.arange(shape[1], dtype=np.int64)[None, :]
    if not np.array_equal(np.sort(array.astype(np.int64), axis=1), np.repeat(expected, shape[0], axis=0)):
        raise ValueError(f"each row of {name} must be a permutation of candidate positions")
    return array.astype(np.intp, copy=False)


def _k_values(k_values: int | Sequence[int]) -> tuple[int, ...]:
    raw_values = (k_values,) if isinstance(k_values, (int, np.integer)) else tuple(k_values)
    if not raw_values:
        raise ValueError("at least one cutoff k is required")
    result: list[int] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("every cutoff k must be an integer")
        cutoff = int(value)
        if cutoff < 1:
            raise ValueError("every cutoff k must be at least 1")
        if cutoff not in result:
            result.append(cutoff)
    return tuple(result)


def _mixed_uint64_keys(
    item_ids: np.ndarray,
    row_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Return the exact modulo-2**64 mixer output used by the archive."""

    seed_value = np.uint64(_seed(seed))
    with np.errstate(over="ignore"):
        mixed = (
            seed_value
            ^ (row_ids[:, None] * GOLDEN)
            ^ (item_ids * np.uint64(int(GOLDEN) >> 1))
        )
        mixed = (mixed ^ (mixed >> 33)) * MIX_1
        mixed = (mixed ^ (mixed >> 33)) * MIX_2
        mixed = mixed ^ (mixed >> 33)
    return mixed.astype(np.uint64, copy=False)


def stable_score_order(
    scores: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Sort scores descending, preserving input position for exact ties.

    This function exposes the positional baseline deliberately. It should not be
    treated as a fair tie policy when candidate construction privileges a slot.
    """

    matrix = _score_matrix(scores)
    return np.argsort(-matrix, axis=1, kind="stable")


def archived_float32_keys(
    item_ids: np.ndarray | Sequence[Sequence[int]],
    row_ids: np.ndarray | Sequence[int] | None = None,
    seed: int = ARCHIVED_SEED,
) -> np.ndarray:
    """Reproduce the historical normalized ``float32`` secondary keys exactly.

    The float conversion is intentional historical behavior, including its loss
    of precision. Use :func:`hardened_uint64_keys` for new evaluations.
    """

    raw_items = np.asarray(item_ids)
    if raw_items.ndim != 2 or raw_items.shape[0] == 0 or raw_items.shape[1] == 0:
        raise ValueError("item_ids must be a non-empty 2-D matrix")
    items = _item_matrix(raw_items, raw_items.shape)
    rows = _row_id_array(row_ids, items.shape[0])
    mixed = _mixed_uint64_keys(items, rows, seed)
    return (mixed.astype(np.float64) / np.float64(2**64)).astype(np.float32)


def archived_order(
    scores: np.ndarray | Sequence[Sequence[float]],
    item_ids: np.ndarray | Sequence[Sequence[int]],
    row_ids: np.ndarray | Sequence[int] | None = None,
    seed: int = ARCHIVED_SEED,
) -> np.ndarray:
    """Reproduce the archived BPC float32-key, two-pass implementation.

    The first ``argsort`` intentionally leaves NumPy's historical default sort
    kind unchanged; the second pass is stable, exactly as in the retained BPC code.
    This is a reproduction policy, not the recommended evaluation policy.
    """

    score_matrix = _score_matrix(scores)
    items = _item_matrix(item_ids, score_matrix.shape)
    keys = archived_float32_keys(items, row_ids=row_ids, seed=seed)
    key_order = np.argsort(keys, axis=1)
    scores_by_key = np.take_along_axis(score_matrix, key_order, axis=1)
    score_order = np.argsort(-scores_by_key, axis=1, kind="stable")
    return np.take_along_axis(key_order, score_order, axis=1)


def hardened_uint64_keys(
    item_ids: np.ndarray | Sequence[Sequence[int]],
    row_ids: np.ndarray | Sequence[int] | None = None,
    seed: int = ARCHIVED_SEED,
) -> np.ndarray:
    """Return full-width deterministic keys for the normative hardened policy."""

    raw_items = np.asarray(item_ids)
    if raw_items.ndim != 2 or raw_items.shape[0] == 0 or raw_items.shape[1] == 0:
        raise ValueError("item_ids must be a non-empty 2-D matrix")
    items = _item_matrix(raw_items, raw_items.shape)
    rows = _row_id_array(row_ids, items.shape[0])
    return _mixed_uint64_keys(items, rows, seed)


def hardened_order(
    scores: np.ndarray | Sequence[Sequence[float]],
    item_ids: np.ndarray | Sequence[Sequence[int]],
    row_ids: np.ndarray | Sequence[int] | None = None,
    seed: int = ARCHIVED_SEED,
    *,
    tie_keys: np.ndarray | Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Sort by score, full-width key, then item identity.

    ``tie_keys`` is an audit hook for replaying or deliberately injecting key
    collisions. Production callers should leave it unset. Regardless of how keys
    are obtained, item identity is the deterministic final collision fallback.
    Candidate identities must therefore be unique within a row.
    """

    score_matrix = _score_matrix(scores)
    items = _item_matrix(item_ids, score_matrix.shape)
    _require_unique_items(items)
    rows = _row_id_array(row_ids, score_matrix.shape[0])
    if tie_keys is None:
        keys = _mixed_uint64_keys(items, rows, seed)
    else:
        keys = _id_array(tie_keys, score_matrix.shape, "tie_keys")

    # Stable least-significant-to-most-significant passes implement the tuple
    # (-score, uint64_key, item_identity) without converting the uint64 keys.
    identity_order = np.argsort(items, axis=1, kind="stable")
    keys_by_identity = np.take_along_axis(keys, identity_order, axis=1)
    key_suborder = np.argsort(keys_by_identity, axis=1, kind="stable")
    secondary_order = np.take_along_axis(identity_order, key_suborder, axis=1)
    scores_by_secondary = np.take_along_axis(score_matrix, secondary_order, axis=1)
    score_suborder = np.argsort(-scores_by_secondary, axis=1, kind="stable")
    return np.take_along_axis(secondary_order, score_suborder, axis=1)


def randomized_uint64_keys(shape: tuple[int, int], seed: int) -> np.ndarray:
    """Generate seeded independent secondary keys without touching model scores."""

    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("shape must describe a non-empty 2-D matrix")
    generator = np.random.Generator(np.random.PCG64(_seed(seed)))
    return generator.bit_generator.random_raw(int(np.prod(shape))).reshape(shape)


def randomized_order(
    scores: np.ndarray | Sequence[Sequence[float]],
    seed: int = ARCHIVED_SEED,
    *,
    tie_keys: np.ndarray | Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Use seeded random secondary keys to order exact score ties.

    This policy is appropriate for repeated-seed sensitivity analysis. It sorts
    the original score matrix lexicographically and never adds score jitter.
    Streaming callers may supply pre-generated ``uint64`` ``tie_keys``; this
    permits one random stream to be consumed across arbitrary chunk boundaries.
    When keys are supplied, ``seed`` is ignored.
    """

    score_matrix = _score_matrix(scores)
    keys = (
        randomized_uint64_keys(score_matrix.shape, seed)
        if tie_keys is None
        else _id_array(tie_keys, score_matrix.shape, "tie_keys")
    )
    key_order = np.argsort(keys, axis=1, kind="stable")
    scores_by_key = np.take_along_axis(score_matrix, key_order, axis=1)
    score_order = np.argsort(-scores_by_key, axis=1, kind="stable")
    return np.take_along_axis(key_order, score_order, axis=1)


def rank_candidates(
    scores: np.ndarray | Sequence[Sequence[float]],
    item_ids: np.ndarray | Sequence[Sequence[int]] | None = None,
    row_ids: np.ndarray | Sequence[int] | None = None,
    *,
    policy: str = "hardened_uint64",
    seed: int = ARCHIVED_SEED,
) -> np.ndarray:
    """Return candidate-position order under an explicit tie policy.

    ``hardened_uint64`` is the default and recommended policy for new results.
    ``archived_float32`` exists only to reproduce previously calculated
    artifacts. The shorter legacy spellings ``hardened`` and ``archived`` are
    accepted as aliases.
    """

    normalized_policy = _POLICY_ALIASES.get(policy, policy)
    if normalized_policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}; got {policy!r}")
    if normalized_policy == "stable":
        return stable_score_order(scores)
    if normalized_policy == "randomized":
        return randomized_order(scores, seed=seed)
    if item_ids is None:
        raise ValueError(f"item_ids are required for the {normalized_policy!r} policy")
    if normalized_policy == "archived_float32":
        return archived_order(scores, item_ids, row_ids=row_ids, seed=seed)
    return hardened_order(scores, item_ids, row_ids=row_ids, seed=seed)


def positive_ranks(
    order: np.ndarray | Sequence[Sequence[int]],
    item_ids: np.ndarray | Sequence[Sequence[int]],
    positive_item_ids: np.ndarray | Sequence[int],
) -> np.ndarray:
    """Return one-based positive rank for each row.

    A row is rejected unless the positive identity occurs exactly once. This
    catches malformed candidate sets instead of silently selecting one duplicate.
    """

    raw_items = np.asarray(item_ids)
    if raw_items.ndim != 2 or raw_items.shape[0] == 0 or raw_items.shape[1] == 0:
        raise ValueError("item_ids must be a non-empty 2-D matrix")
    items = _item_matrix(raw_items, raw_items.shape)
    ranking = _validated_order(order, items.shape)
    positives = _id_array(positive_item_ids, (items.shape[0],), "positive_item_ids")
    ranked_items = np.take_along_axis(items, ranking, axis=1)
    matches = ranked_items == positives[:, None]
    match_counts = matches.sum(axis=1)
    if np.any(match_counts != 1):
        bad_rows = np.flatnonzero(match_counts != 1)
        preview = ", ".join(str(int(row)) for row in bad_rows[:5])
        raise ValueError(
            "each positive_item_id must occur exactly once in its candidate row; "
            f"invalid row(s): {preview}"
        )
    return np.argmax(matches, axis=1).astype(np.int64) + 1


def per_row_hit_ndcg(
    order: np.ndarray | Sequence[Sequence[int]],
    item_ids: np.ndarray | Sequence[Sequence[int]],
    positive_item_ids: np.ndarray | Sequence[int],
    k_values: int | Sequence[int] = (10,),
) -> dict[str, np.ndarray]:
    """Compute per-row Hit@k and NDCG@k for exactly one positive per row."""

    ranks = positive_ranks(order, item_ids, positive_item_ids)
    metrics: dict[str, np.ndarray] = {}
    for cutoff in _k_values(k_values):
        hits = ranks <= cutoff
        metrics[f"hit@{cutoff}"] = hits.astype(np.int8)
        metrics[f"ndcg@{cutoff}"] = np.where(
            hits,
            1.0 / np.log2(ranks.astype(np.float64) + 1.0),
            0.0,
        )
    return metrics


def evaluate_rows(
    scores: np.ndarray | Sequence[Sequence[float]],
    item_ids: np.ndarray | Sequence[Sequence[int]],
    positive_item_ids: np.ndarray | Sequence[int],
    k_values: int | Sequence[int] = (10,),
    row_ids: np.ndarray | Sequence[int] | None = None,
    *,
    policy: str = "hardened_uint64",
    seed: int = ARCHIVED_SEED,
) -> dict[str, np.ndarray]:
    """Rank candidates and return per-row metrics under the selected policy."""

    order = rank_candidates(
        scores,
        item_ids,
        row_ids,
        policy=policy,
        seed=seed,
    )
    return per_row_hit_ndcg(order, item_ids, positive_item_ids, k_values)


def expected_hit_at_k(a: int, t: int, k: int) -> float:
    """Analytic expected Hit@k for one positive uniformly ordered in a tie block.

    ``a`` is the number of candidates with strictly higher scores and ``t`` is
    the complete tie-block size, including the positive.
    """

    higher, tie_size, cutoff = _analytic_inputs(a, t, k)
    qualifying_ranks = max(0, min(tie_size, cutoff - higher))
    return float(qualifying_ranks / tie_size)


def expected_ndcg_at_k(a: int, t: int, k: int) -> float:
    """Analytic expected NDCG@k for one positive in a uniform tie block."""

    higher, tie_size, cutoff = _analytic_inputs(a, t, k)
    last_rank = min(higher + tie_size, cutoff)
    if last_rank <= higher:
        return 0.0
    ranks = np.arange(higher + 1, last_rank + 1, dtype=np.float64)
    return float(np.sum(1.0 / np.log2(ranks + 1.0)) / tie_size)


def _analytic_inputs(a: int, t: int, k: int) -> tuple[int, int, int]:
    values = (a, t, k)
    names = ("a", "t", "k")
    for value, name in zip(values, names):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    higher, tie_size, cutoff = (int(value) for value in values)
    if higher < 0:
        raise ValueError("a must be non-negative")
    if tie_size < 1:
        raise ValueError("t must be at least 1 and includes the positive")
    if cutoff < 1:
        raise ValueError("k must be at least 1")
    return higher, tie_size, cutoff


def expected_one_positive_metrics(
    a: int,
    t: int,
    k_values: int | Sequence[int] = (10,),
) -> dict[str, float]:
    """Return analytic expected Hit/NDCG values for one tied positive."""

    metrics: dict[str, float] = {}
    for cutoff in _k_values(k_values):
        metrics[f"hit@{cutoff}"] = expected_hit_at_k(a, t, cutoff)
        metrics[f"ndcg@{cutoff}"] = expected_ndcg_at_k(a, t, cutoff)
    return metrics


def tie_diagnostics(
    scores: np.ndarray | Sequence[Sequence[float]],
    *,
    positive_positions: np.ndarray | Sequence[int] | None = None,
    k: int | None = None,
) -> dict[str, int | float]:
    """Summarize exact-score tie prevalence and optional positive/cutoff exposure.

    ``positive_positions`` refers to positions in the input score matrix. When
    ``k`` is supplied, the report distinguishes any tie reaching the first ``k``
    ranks from a tie block that actually crosses the top-k membership boundary.
    """

    matrix = _score_matrix(scores)
    n_rows, n_candidates = matrix.shape
    positions: np.ndarray | None = None
    if positive_positions is not None:
        raw_positions = np.asarray(positive_positions)
        if raw_positions.shape != (n_rows,) or not np.issubdtype(raw_positions.dtype, np.integer):
            raise ValueError(f"positive_positions must be an integer vector of shape ({n_rows},)")
        positions = raw_positions.astype(np.int64)
        if np.any((positions < 0) | (positions >= n_candidates)):
            raise ValueError("positive_positions contains an out-of-range position")

    cutoff: int | None = None
    if k is not None:
        cutoff = _k_values(k)[0]

    rows_with_ties = 0
    tied_candidates = 0
    maximum_tie_size = 1
    positives_in_ties = 0
    top_k_tie_rows = 0
    cutoff_tie_rows = 0
    cutoff_crossing_rows = 0

    for row_number, row in enumerate(matrix):
        unique_scores, inverse, counts = np.unique(row, return_inverse=True, return_counts=True)
        del unique_scores
        sizes = counts[inverse]
        tied_mask = sizes > 1
        if np.any(tied_mask):
            rows_with_ties += 1
        tied_candidates += int(np.sum(tied_mask))
        maximum_tie_size = max(maximum_tie_size, int(counts.max()))
        if positions is not None and tied_mask[positions[row_number]]:
            positives_in_ties += 1

        if cutoff is not None:
            effective_k = min(cutoff, n_candidates)
            descending = np.sort(row)[::-1]
            kth_score = descending[effective_k - 1]
            greater = int(np.sum(row > kth_score))
            equal = int(np.sum(row == kth_score))
            if equal > 1:
                cutoff_tie_rows += 1
            if greater < effective_k < greater + equal:
                cutoff_crossing_rows += 1

            # A tied group reaches top-k if its best possible rank is <= k.
            reaches_top_k = False
            for candidate_score, group_size in zip(row, sizes):
                if group_size > 1 and int(np.sum(row > candidate_score)) < effective_k:
                    reaches_top_k = True
                    break
            if reaches_top_k:
                top_k_tie_rows += 1

    report: dict[str, int | float] = {
        "row_count": n_rows,
        "candidate_count": n_candidates,
        "rows_with_ties": rows_with_ties,
        "row_tie_fraction": rows_with_ties / n_rows,
        "tied_candidates": tied_candidates,
        "tied_candidate_fraction": tied_candidates / (n_rows * n_candidates),
        "maximum_tie_size": maximum_tie_size,
    }
    if positions is not None:
        report["positives_in_ties"] = positives_in_ties
        report["positive_tie_fraction"] = positives_in_ties / n_rows
    if cutoff is not None:
        report["top_k_tie_rows"] = top_k_tie_rows
        report["top_k_tie_fraction"] = top_k_tie_rows / n_rows
        report["cutoff_tie_rows"] = cutoff_tie_rows
        report["cutoff_tie_fraction"] = cutoff_tie_rows / n_rows
        report["cutoff_crossing_tie_rows"] = cutoff_crossing_rows
        report["cutoff_crossing_tie_fraction"] = cutoff_crossing_rows / n_rows
    return report


def key_collision_counts(
    keys: np.ndarray | Sequence[int] | Sequence[Sequence[int]],
    item_ids: np.ndarray | Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Count excess distinct identities sharing a key, independently per row.

    A group of three different items with the same key contributes two
    collisions. Repeated occurrences of the same identity do not count as a key
    collision. If ``item_ids`` is omitted, every occurrence is treated as a
    distinct candidate.
    """

    key_array = np.asarray(keys)
    was_vector = key_array.ndim == 1
    if was_vector:
        key_array = key_array[None, :]
    if key_array.ndim != 2 or key_array.shape[1] == 0:
        raise ValueError("keys must be a non-empty vector or 2-D matrix")
    if np.issubdtype(key_array.dtype, np.floating) and np.isnan(key_array).any():
        raise ValueError("keys must not contain NaN")

    identities: np.ndarray | None = None
    if item_ids is not None:
        raw_items = np.asarray(item_ids)
        if was_vector and raw_items.ndim == 1:
            raw_items = raw_items[None, :]
        identities = _id_array(raw_items, key_array.shape, "item_ids")

    result = np.zeros(key_array.shape[0], dtype=np.int64)
    for row_number, row_keys in enumerate(key_array):
        _, inverse = np.unique(row_keys, return_inverse=True)
        for group in range(int(inverse.max()) + 1):
            members = inverse == group
            distinct_count = (
                int(np.sum(members))
                if identities is None
                else int(np.unique(identities[row_number, members]).size)
            )
            result[row_number] += max(0, distinct_count - 1)
    return result


def count_key_collisions(
    keys: np.ndarray | Sequence[int] | Sequence[Sequence[int]],
    item_ids: np.ndarray | Sequence[int] | Sequence[Sequence[int]] | None = None,
) -> int:
    """Return the total collision count across rows."""

    return int(key_collision_counts(keys, item_ids).sum())


def apply_row_permutations(
    values: np.ndarray | Sequence[Sequence[object]],
    permutations: np.ndarray | Sequence[Sequence[int]],
) -> np.ndarray:
    """Apply one candidate-position permutation per row."""

    array = np.asarray(values)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("values must be a non-empty 2-D matrix")
    order = _validated_order(permutations, array.shape, "permutations")
    return np.take_along_axis(array, order, axis=1)


def compare_rankings(
    order_a: np.ndarray | Sequence[Sequence[int]],
    item_ids_a: np.ndarray | Sequence[Sequence[int]],
    order_b: np.ndarray | Sequence[Sequence[int]],
    item_ids_b: np.ndarray | Sequence[Sequence[int]],
    *,
    k: int | None = None,
) -> dict[str, int | float]:
    """Compare rankings by item identity, even when inputs were permuted."""

    raw_a = np.asarray(item_ids_a)
    raw_b = np.asarray(item_ids_b)
    if raw_a.ndim != 2 or raw_a.shape[0] == 0 or raw_a.shape[1] == 0:
        raise ValueError("item_ids_a must be a non-empty 2-D matrix")
    if raw_b.shape != raw_a.shape:
        raise ValueError("item_ids_a and item_ids_b must have the same shape")
    items_a = _item_matrix(raw_a, raw_a.shape)
    items_b = _item_matrix(raw_b, raw_a.shape)
    if not np.array_equal(np.sort(items_a, axis=1), np.sort(items_b, axis=1)):
        raise ValueError("the two inputs must contain the same candidate identities per row")
    ranking_a = _validated_order(order_a, raw_a.shape, "order_a")
    ranking_b = _validated_order(order_b, raw_a.shape, "order_b")
    ranked_a = np.take_along_axis(items_a, ranking_a, axis=1)
    ranked_b = np.take_along_axis(items_b, ranking_b, axis=1)

    n_rows, n_candidates = raw_a.shape
    equal_positions = ranked_a == ranked_b
    exact_rows = np.all(equal_positions, axis=1)
    report: dict[str, int | float] = {
        "row_count": n_rows,
        "candidate_count": n_candidates,
        "exact_match_rows": int(np.sum(exact_rows)),
        "exact_match_fraction": float(np.mean(exact_rows)),
        "differing_positions": int(np.sum(~equal_positions)),
        "differing_position_fraction": float(np.mean(~equal_positions)),
    }
    if k is not None:
        cutoff = min(_k_values(k)[0], n_candidates)
        top_a = ranked_a[:, :cutoff]
        top_b = ranked_b[:, :cutoff]
        ordered_matches = np.all(top_a == top_b, axis=1)
        set_matches = np.all(np.sort(top_a, axis=1) == np.sort(top_b, axis=1), axis=1)
        report["top_k"] = cutoff
        report["top_k_order_match_rows"] = int(np.sum(ordered_matches))
        report["top_k_order_match_fraction"] = float(np.mean(ordered_matches))
        report["top_k_set_match_rows"] = int(np.sum(set_matches))
        report["top_k_set_match_fraction"] = float(np.mean(set_matches))
    return report


def permutation_audit(
    scores: np.ndarray | Sequence[Sequence[float]],
    item_ids: np.ndarray | Sequence[Sequence[int]],
    permutations: np.ndarray | Sequence[Sequence[int]],
    row_ids: np.ndarray | Sequence[int] | None = None,
    *,
    policy: str = "hardened_uint64",
    seed: int = ARCHIVED_SEED,
    k: int | None = None,
) -> dict[str, int | float]:
    """Rank original/permuted inputs and compare the resulting item rankings."""

    score_matrix = _score_matrix(scores)
    items = _item_matrix(item_ids, score_matrix.shape)
    permuted_scores = apply_row_permutations(score_matrix, permutations)
    permuted_items = apply_row_permutations(items, permutations)
    original_order = rank_candidates(
        score_matrix,
        items,
        row_ids,
        policy=policy,
        seed=seed,
    )
    permuted_order = rank_candidates(
        permuted_scores,
        permuted_items,
        row_ids,
        policy=policy,
        seed=seed,
    )
    return compare_rankings(
        original_order,
        items,
        permuted_order,
        permuted_items,
        k=k,
    )
