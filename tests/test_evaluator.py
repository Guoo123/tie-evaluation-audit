"""Regression and invariance tests for the public tie-audit evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_frozen_rows import randomized_metric_runs, validate_output_target

from tie_audit import (
    apply_row_permutations,
    archived_float32_keys,
    archived_order,
    compare_rankings,
    count_key_collisions,
    expected_hit_at_k,
    expected_ndcg_at_k,
    expected_one_positive_metrics,
    hardened_order,
    hardened_uint64_keys,
    key_collision_counts,
    per_row_hit_ndcg,
    permutation_audit,
    randomized_order,
    randomized_uint64_keys,
    stable_score_order,
    tie_diagnostics,
)


def test_one_positive_plus_thirty_all_tied_analytic_formulas() -> None:
    """The positive is uniform over ranks 1..31 under an unbiased tie rule."""

    expected_hit = 10.0 / 31.0
    expected_ndcg = float(
        np.sum(1.0 / np.log2(np.arange(1, 11, dtype=np.float64) + 1.0)) / 31.0
    )

    assert expected_hit_at_k(a=0, t=31, k=10) == expected_hit
    assert expected_ndcg_at_k(a=0, t=31, k=10) == expected_ndcg
    metrics = expected_one_positive_metrics(a=0, t=31, k_values=(1, 10, 31))
    assert metrics["hit@1"] == 1.0 / 31.0
    assert metrics["hit@10"] == expected_hit
    assert metrics["hit@31"] == 1.0
    assert metrics["ndcg@10"] == expected_ndcg

    # If ten candidates score strictly higher, no member of the following tie
    # block can enter the first ten positions.
    assert expected_hit_at_k(a=10, t=21, k=10) == 0.0
    assert expected_ndcg_at_k(a=10, t=21, k=10) == 0.0


def test_stable_score_only_order_preserves_positive_first_and_last() -> None:
    n_candidates = 31
    scores = np.ones((2, n_candidates), dtype=np.float32)
    items = np.vstack(
        (
            np.arange(100, 100 + n_candidates),
            np.arange(200, 200 + n_candidates),
        )
    )
    positives = np.array([items[0, 0], items[1, -1]])

    order = stable_score_order(scores)
    np.testing.assert_array_equal(
        order,
        np.repeat(np.arange(n_candidates)[None, :], 2, axis=0),
    )
    metrics = per_row_hit_ndcg(order, items, positives, k_values=(1, 10, 31))

    np.testing.assert_array_equal(metrics["hit@1"], np.array([1, 0]))
    np.testing.assert_array_equal(metrics["hit@10"], np.array([1, 0]))
    np.testing.assert_array_equal(metrics["hit@31"], np.array([1, 1]))
    np.testing.assert_allclose(metrics["ndcg@31"], [1.0, 1.0 / np.log2(32.0)])


def test_hardened_ranking_is_invariant_to_input_permutations() -> None:
    scores = np.array(
        [
            [3.0, 2.0, 2.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.5, 0.5, -1.0],
            [8.0, 4.0, 4.0, 4.0, 4.0, 2.0],
        ]
    )
    items = np.array(
        [
            [90, 11, 72, 40, 8, 63],
            [101, 107, 103, 102, 109, 105],
            [210, 205, 219, 202, 217, 201],
        ],
        dtype=np.int64,
    )
    row_ids = np.array([700, 701, 702], dtype=np.int64)
    permutations = np.array(
        [
            [5, 1, 4, 0, 3, 2],
            [2, 5, 0, 4, 1, 3],
            [3, 0, 5, 2, 4, 1],
        ]
    )

    report = permutation_audit(
        scores,
        items,
        permutations,
        row_ids,
        policy="hardened_uint64",
        seed=19,
        k=3,
    )
    assert report["exact_match_fraction"] == 1.0
    assert report["differing_positions"] == 0
    assert report["top_k_order_match_fraction"] == 1.0
    assert report["top_k_set_match_fraction"] == 1.0


def test_all_policies_are_equivalent_without_score_ties() -> None:
    scores = np.array(
        [
            [0.3, 9.0, -2.0, 4.0, 1.0],
            [7.0, 2.0, 6.0, 1.0, 4.0],
        ],
        dtype=np.float64,
    )
    items = np.array([[8, 6, 4, 2, 0], [31, 37, 39, 33, 35]], dtype=np.int64)
    row_ids = np.array([91, 92], dtype=np.int64)

    stable = stable_score_order(scores)
    archived = archived_order(scores, items, row_ids, seed=123)
    hardened = hardened_order(scores, items, row_ids, seed=123)
    randomized = randomized_order(scores, seed=123)

    np.testing.assert_array_equal(archived, stable)
    np.testing.assert_array_equal(hardened, stable)
    np.testing.assert_array_equal(randomized, stable)


def test_archived_mixer_and_two_pass_order_are_exact_and_deterministic() -> None:
    items = np.array(
        [[0, 1, 2, 2**32 + 7], [19, 3, 41, 8]],
        dtype=np.uint64,
    )
    row_ids = np.array([5, 9001], dtype=np.uint64)
    seed = 20260316

    # Independent transcription of the archived arithmetic. In particular, the
    # division occurs in float64 and the result is then rounded to float32.
    golden = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        raw = (
            np.uint64(seed)
            ^ (row_ids[:, None] * golden)
            ^ (items * np.uint64(int(golden) >> 1))
        )
        raw = (raw ^ (raw >> 33)) * np.uint64(0xFF51AFD7ED558CCD)
        raw = (raw ^ (raw >> 33)) * np.uint64(0xC4CEB9FE1A85EC53)
        raw = raw ^ (raw >> 33)
    expected_archived = (raw.astype(np.float64) / np.float64(2**64)).astype(np.float32)

    archived_keys = archived_float32_keys(items, row_ids, seed=seed)
    hardened_keys = hardened_uint64_keys(items, row_ids, seed=seed)
    assert archived_keys.dtype == np.float32
    assert hardened_keys.dtype == np.uint64
    np.testing.assert_array_equal(archived_keys, expected_archived)
    np.testing.assert_array_equal(hardened_keys, raw)

    scores = np.array([[1, 1, 1, 1], [5, 2, 2, 2]], dtype=np.float32)
    first = archived_order(scores, items, row_ids, seed=seed)
    second = archived_order(scores.copy(), items.copy(), row_ids.copy(), seed=seed)
    key_order = np.argsort(expected_archived, axis=1)
    scores_by_key = np.take_along_axis(scores, key_order, axis=1)
    score_order = np.argsort(-scores_by_key, axis=1, kind="stable")
    expected_order = np.take_along_axis(key_order, score_order, axis=1)
    np.testing.assert_array_equal(first, expected_order)
    np.testing.assert_array_equal(first, second)


def test_seeded_randomized_ties_converge_to_analytic_expectation() -> None:
    n_rows = 30_000
    n_candidates = 31
    scores = np.ones((n_rows, n_candidates), dtype=np.float32)
    original_scores = scores.copy()
    items = np.repeat(np.arange(n_candidates, dtype=np.int64)[None, :], n_rows, axis=0)
    positives = np.zeros(n_rows, dtype=np.int64)

    keys_a = randomized_uint64_keys((4, n_candidates), seed=8675309)
    keys_b = randomized_uint64_keys((4, n_candidates), seed=8675309)
    np.testing.assert_array_equal(keys_a, keys_b)

    order = randomized_order(scores, seed=8675309)
    np.testing.assert_array_equal(scores, original_scores)  # no score jitter
    metrics = per_row_hit_ndcg(order, items, positives, k_values=10)
    analytic = expected_one_positive_metrics(a=0, t=31, k_values=10)

    assert abs(float(metrics["hit@10"].mean()) - analytic["hit@10"]) < 0.01
    assert abs(float(metrics["ndcg@10"].mean()) - analytic["ndcg@10"]) < 0.01


def test_hardened_identity_fallback_handles_deliberate_key_collisions() -> None:
    scores = np.ones((1, 4), dtype=np.float64)
    items = np.array([[30, 10, 20, 40]], dtype=np.int64)
    colliding_keys = np.array([[9, 9, 9, 2]], dtype=np.uint64)

    order_a = hardened_order(scores, items, row_ids=[7], tie_keys=colliding_keys)
    ranked_a = np.take_along_axis(items, order_a, axis=1)
    np.testing.assert_array_equal(ranked_a, [[40, 10, 20, 30]])
    np.testing.assert_array_equal(key_collision_counts(colliding_keys, items), [2])
    assert count_key_collisions(colliding_keys, items) == 2
    assert count_key_collisions([5, 5], [17, 17]) == 0

    permutation = np.array([[2, 3, 0, 1]])
    permuted_scores = apply_row_permutations(scores, permutation)
    permuted_items = apply_row_permutations(items, permutation)
    permuted_keys = apply_row_permutations(colliding_keys, permutation)
    order_b = hardened_order(
        permuted_scores,
        permuted_items,
        row_ids=[7],
        tie_keys=permuted_keys,
    )
    report = compare_rankings(order_a, items, order_b, permuted_items, k=3)
    assert report["exact_match_fraction"] == 1.0
    assert report["top_k_order_match_fraction"] == 1.0


def test_real_archived_float32_collision_can_reintroduce_row_order_dependence() -> None:
    """The historical float32 conversion is reproducible but not a strict guarantee."""

    # These two item identities collide after the archived uint64 mixer is
    # normalized and rounded to float32 for row 0 and seed 20260316.
    items_a = np.array([[28584, 95915]], dtype=np.int64)
    items_b = items_a[:, ::-1].copy()
    scores = np.ones((1, 2), dtype=np.float32)

    keys_a = archived_float32_keys(items_a, row_ids=[0], seed=20260316)
    keys_b = archived_float32_keys(items_b, row_ids=[0], seed=20260316)
    assert keys_a[0, 0] == keys_a[0, 1]
    assert keys_b[0, 0] == keys_b[0, 1]
    assert count_key_collisions(keys_a, items_a) == 1

    archived_a = archived_order(scores, items_a, row_ids=[0], seed=20260316)
    archived_b = archived_order(scores, items_b, row_ids=[0], seed=20260316)
    archived_report = compare_rankings(archived_a, items_a, archived_b, items_b)
    assert archived_report["exact_match_fraction"] == 0.0

    hardened_a = hardened_order(scores, items_a, row_ids=[0], seed=20260316)
    hardened_b = hardened_order(scores, items_b, row_ids=[0], seed=20260316)
    hardened_report = compare_rankings(hardened_a, items_a, hardened_b, items_b)
    assert hardened_report["exact_match_fraction"] == 1.0


def test_tie_diagnostics_distinguish_ties_from_cutoff_crossings() -> None:
    scores = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 4.0, 3.0],
        ]
    )
    diagnostics = tie_diagnostics(scores, positive_positions=[0, 3, 3], k=2)

    assert diagnostics["rows_with_ties"] == 2
    assert diagnostics["row_tie_fraction"] == 2.0 / 3.0
    assert diagnostics["tied_candidates"] == 5
    assert diagnostics["maximum_tie_size"] == 3
    assert diagnostics["positives_in_ties"] == 1
    assert diagnostics["positive_tie_fraction"] == 1.0 / 3.0
    assert diagnostics["top_k_tie_rows"] == 2
    assert diagnostics["cutoff_tie_rows"] == 2
    assert diagnostics["cutoff_crossing_tie_rows"] == 2


def test_randomized_frozen_row_metrics_are_chunk_size_invariant() -> None:
    n_rows = 23
    n_candidates = 9
    candidate_ids = (
        np.arange(n_rows * n_candidates, dtype=np.int64).reshape(n_rows, n_candidates)
        + 100
    )
    scores = np.tile(
        np.array([3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        (n_rows, 1),
    )
    positive_positions = np.arange(n_rows) % n_candidates
    positive_item_ids = candidate_ids[np.arange(n_rows), positive_positions]

    one_row_chunks = randomized_metric_runs(
        scores,
        candidate_ids,
        positive_item_ids,
        k=4,
        repetitions=5,
        random_seed=771,
        chunk_size=1,
    )
    uneven_chunks = randomized_metric_runs(
        scores,
        candidate_ids,
        positive_item_ids,
        k=4,
        repetitions=5,
        random_seed=771,
        chunk_size=7,
    )
    single_chunk = randomized_metric_runs(
        scores,
        candidate_ids,
        positive_item_ids,
        k=4,
        repetitions=5,
        random_seed=771,
        chunk_size=n_rows,
    )

    assert one_row_chunks == uneven_chunks == single_chunk


def test_output_guard_requires_force_and_never_allows_input_clobber(tmp_path) -> None:
    input_path = tmp_path / "frozen_rows.npz"
    output_path = tmp_path / "audit.json"
    input_path.write_bytes(b"frozen payload sentinel")

    validate_output_target(input_path, output_path, force=False)
    output_path.write_text("existing report", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        validate_output_target(input_path, output_path, force=False)
    validate_output_target(input_path, output_path, force=True)

    with pytest.raises(SystemExit, match="must not refer"):
        validate_output_target(input_path, input_path, force=False)
    with pytest.raises(SystemExit, match="must not refer"):
        validate_output_target(input_path, input_path, force=True)
