#!/usr/bin/env python3
"""Run the paper's all-tied counterexample without external data."""

from __future__ import annotations

import json

import numpy as np

from tie_audit import (
    evaluate_rows,
    expected_one_positive_metrics,
    rank_candidates,
    positive_ranks,
)


K = 10
N_CANDIDATES = 31


def aggregate(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(values.mean()) for name, values in metrics.items()}


def main() -> None:
    item_ids = np.arange(N_CANDIDATES, dtype=np.int64)[None, :]
    scores = np.ones((1, N_CANDIDATES), dtype=np.float32)
    positive_ids = np.array([0], dtype=np.int64)
    row_ids = np.array([7], dtype=np.int64)

    stable_first = aggregate(
        evaluate_rows(
            scores,
            item_ids,
            positive_ids,
            K,
            row_ids,
            policy="stable",
        )
    )

    reverse = np.arange(N_CANDIDATES - 1, -1, -1)
    stable_last = aggregate(
        evaluate_rows(
            scores[:, reverse],
            item_ids[:, reverse],
            positive_ids,
            K,
            row_ids,
            policy="stable",
        )
    )

    analytic = expected_one_positive_metrics(a=0, t=N_CANDIDATES, k_values=K)

    n_random_rows = 20_000
    random_items = np.repeat(item_ids, n_random_rows, axis=0)
    random_scores = np.repeat(scores, n_random_rows, axis=0)
    random_positives = np.zeros(n_random_rows, dtype=np.int64)
    randomized = aggregate(
        evaluate_rows(
            random_scores,
            random_items,
            random_positives,
            K,
            policy="randomized",
            seed=2026,
        )
    )

    single_realizations: dict[str, dict[str, float | int]] = {}
    for policy in ("archived_float32", "hardened_uint64"):
        order = rank_candidates(
            scores,
            item_ids,
            row_ids,
            policy=policy,
            seed=20260316,
        )
        policy_metrics = aggregate(
            evaluate_rows(
                scores,
                item_ids,
                positive_ids,
                K,
                row_ids,
                policy=policy,
                seed=20260316,
            )
        )
        policy_metrics["positive_rank"] = int(
            positive_ranks(order, item_ids, positive_ids)[0]
        )
        single_realizations[policy] = policy_metrics

    report = {
        "frozen_row": {
            "relevant_items": 1,
            "sampled_negatives": 30,
            "all_primary_scores_equal": True,
            "k": K,
        },
        "stable_positive_first": stable_first,
        "stable_positive_last": stable_last,
        "uniform_tie_order_analytic_expectation": analytic,
        "uniform_tie_order_monte_carlo": {
            "rows": n_random_rows,
            **randomized,
        },
        "deterministic_keyed_single_realizations": single_realizations,
    }

    assert stable_first["hit@10"] == 1.0
    assert stable_first["ndcg@10"] == 1.0
    assert stable_last["hit@10"] == 0.0
    assert stable_last["ndcg@10"] == 0.0
    assert abs(analytic["hit@10"] - 10 / 31) < 1e-12
    assert abs(analytic["ndcg@10"] - 0.1466) < 5e-5
    assert abs(randomized["hit@10"] - analytic["hit@10"]) < 0.015
    assert abs(randomized["ndcg@10"] - analytic["ndcg@10"]) < 0.01

    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nPASS: row construction alone changes stable metrics; the analytic and randomized expectations agree.")


if __name__ == "__main__":
    main()
