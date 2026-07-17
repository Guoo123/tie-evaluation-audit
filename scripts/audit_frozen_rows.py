#!/usr/bin/env python3
"""Audit stable, archived, hardened, analytic, and randomized tie policies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tie_audit import (
    archived_float32_keys,
    count_key_collisions,
    evaluate_rows,
    hardened_uint64_keys,
    permutation_audit,
    positive_ranks,
    randomized_order,
    tie_diagnostics,
)


REQUIRED_ARRAYS = ("candidate_ids", "primary_scores", "positive_item_ids", "user_ids")


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def mean_metrics(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(values.mean()) for name, values in metrics.items()}


def validate_output_target(input_path: Path, output_path: Path, force: bool) -> None:
    """Reject input clobbering and require explicit consent for replacement."""

    input_resolved = input_path.resolve()
    output_resolved = output_path.resolve()
    same_file = input_resolved == output_resolved
    if input_path.exists() and output_path.exists():
        try:
            same_file = same_file or input_path.samefile(output_path)
        except OSError:
            pass
    if same_file:
        raise SystemExit("--output must not refer to the --input payload")
    if output_path.exists() and output_path.is_dir():
        raise SystemExit("--output must be a file path, not an existing directory")
    if output_path.exists() and not force:
        raise SystemExit("--output already exists; pass --force to replace it")


def randomized_metric_runs(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    positive_item_ids: np.ndarray,
    k: int,
    repetitions: int,
    random_seed: int,
    chunk_size: int,
) -> list[dict[str, float]]:
    """Evaluate randomized ties identically for every positive chunk size.

    Each repetition owns one PCG64 stream. Chunks consume consecutive raw keys
    from that stream in global row-major order. Positive ranks are assembled for
    the complete run before aggregation, avoiding chunk-dependent float sums.
    """

    n_rows = len(scores)
    runs: list[dict[str, float]] = []
    for repetition in range(repetitions):
        generator = np.random.Generator(np.random.PCG64(random_seed + repetition))
        ranks = np.empty(n_rows, dtype=np.int64)
        for start in range(0, n_rows, chunk_size):
            end = min(start + chunk_size, n_rows)
            block_scores = scores[start:end]
            tie_keys = generator.bit_generator.random_raw(block_scores.size).reshape(
                block_scores.shape
            )
            order = randomized_order(block_scores, tie_keys=tie_keys)
            ranks[start:end] = positive_ranks(
                order,
                candidate_ids[start:end],
                positive_item_ids[start:end],
            )

        hits = ranks <= k
        ndcg = np.where(
            hits,
            1.0 / np.log2(ranks.astype(np.float64) + 1.0),
            0.0,
        )
        runs.append(
            {
                f"hit@{k}": float(hits.mean()),
                f"ndcg@{k}": float(ndcg.mean()),
            }
        )
    return runs


def analytic_means(
    scores: np.ndarray,
    positive_positions: np.ndarray,
    k: int,
    chunk_size: int,
) -> dict[str, float]:
    hit_sum = 0.0
    ndcg_sum = 0.0
    rank_axis = np.arange(1, scores.shape[1] + 1, dtype=np.int64)[None, :]
    discounts = 1.0 / np.log2(rank_axis.astype(np.float64) + 1.0)

    for start in range(0, len(scores), chunk_size):
        end = min(start + chunk_size, len(scores))
        block = scores[start:end]
        positions = positive_positions[start:end]
        positive_scores = block[np.arange(len(block)), positions]
        higher = np.sum(block > positive_scores[:, None], axis=1)
        tied = np.sum(block == positive_scores[:, None], axis=1)
        qualifying = np.maximum(0, np.minimum(tied, k - higher))
        hit_sum += float(np.sum(qualifying / tied))

        valid = (
            (rank_axis > higher[:, None])
            & (rank_axis <= (higher + tied)[:, None])
            & (rank_axis <= k)
        )
        ndcg_sum += float(np.sum(np.sum(valid * discounts, axis=1) / tied))

    return {"hit@k": hit_sum / len(scores), "ndcg@k": ndcg_sum / len(scores)}


def full_permutation_audit(
    scores: np.ndarray,
    candidate_ids: np.ndarray,
    user_ids: np.ndarray,
    policy: str,
    seed: int,
    permutation_seed: int,
    k: int,
    chunk_size: int,
) -> dict[str, float | int]:
    generator = np.random.default_rng(permutation_seed)
    totals = {
        "row_count": 0,
        "exact_match_rows": 0,
        "differing_positions": 0,
        "total_positions": 0,
        "top_k_order_match_rows": 0,
        "top_k_set_match_rows": 0,
    }

    for start in range(0, len(scores), chunk_size):
        end = min(start + chunk_size, len(scores))
        block_size = end - start
        random_keys = generator.random((block_size, scores.shape[1]))
        permutations = np.argsort(random_keys, axis=1, kind="stable")
        report = permutation_audit(
            scores[start:end],
            candidate_ids[start:end],
            permutations,
            user_ids[start:end],
            policy=policy,
            seed=seed,
            k=k,
        )
        totals["row_count"] += int(report["row_count"])
        totals["exact_match_rows"] += int(report["exact_match_rows"])
        totals["differing_positions"] += int(report["differing_positions"])
        totals["total_positions"] += int(report["row_count"]) * int(report["candidate_count"])
        totals["top_k_order_match_rows"] += int(report["top_k_order_match_rows"])
        totals["top_k_set_match_rows"] += int(report["top_k_set_match_rows"])

    rows = totals["row_count"]
    return {
        **totals,
        "exact_match_fraction": totals["exact_match_rows"] / rows,
        "differing_position_fraction": totals["differing_positions"] / totals["total_positions"],
        "top_k_order_match_fraction": totals["top_k_order_match_rows"] / rows,
        "top_k_set_match_fraction": totals["top_k_set_match_rows"] / rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Frozen-row NPZ payload")
    parser.add_argument("--dataset", required=True, help="Neutral dataset label for the report")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--archived-seed", type=int, required=True)
    parser.add_argument("--random-repeats", type=int, default=32)
    parser.add_argument("--random-seed", type=int, default=73001)
    parser.add_argument("--permutation-seed", type=int, default=91001)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file (never permits overwriting the input)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1 or args.random_repeats < 1 or args.chunk_size < 1:
        raise SystemExit("k, random-repeats, and chunk-size must be positive")
    validate_output_target(args.input, args.output, args.force)

    with np.load(args.input, allow_pickle=False) as payload:
        missing = [name for name in REQUIRED_ARRAYS if name not in payload]
        if missing:
            raise SystemExit(f"missing required NPZ arrays: {', '.join(missing)}")
        candidate_ids = np.asarray(payload["candidate_ids"])
        primary_scores = np.asarray(payload["primary_scores"])
        positive_item_ids = np.asarray(payload["positive_item_ids"])
        user_ids = np.asarray(payload["user_ids"])

    if candidate_ids.ndim != 2 or primary_scores.shape != candidate_ids.shape:
        raise SystemExit("candidate_ids and primary_scores must have the same 2-D shape")
    n_rows, n_candidates = candidate_ids.shape
    if positive_item_ids.shape != (n_rows,) or user_ids.shape != (n_rows,):
        raise SystemExit("positive_item_ids and user_ids must have one value per row")
    if not np.isfinite(primary_scores).all():
        raise SystemExit("primary_scores must be finite")

    positive_matches = candidate_ids == positive_item_ids[:, None]
    match_counts = positive_matches.sum(axis=1)
    if np.any(match_counts != 1):
        bad = np.flatnonzero(match_counts != 1)[:10].tolist()
        raise SystemExit(f"positive item must occur exactly once; invalid rows: {bad}")
    positive_positions = np.argmax(positive_matches, axis=1)

    metrics: dict[str, Any] = {}
    for policy in ("stable", "archived_float32", "hardened_uint64"):
        policy_sums = {f"hit@{args.k}": 0.0, f"ndcg@{args.k}": 0.0}
        for start in range(0, n_rows, args.chunk_size):
            end = min(start + args.chunk_size, n_rows)
            block = evaluate_rows(
                primary_scores[start:end],
                candidate_ids[start:end],
                positive_item_ids[start:end],
                args.k,
                user_ids[start:end],
                policy=policy,
                seed=args.archived_seed,
            )
            for name, values in block.items():
                policy_sums[name] += float(values.sum())
        metrics[policy] = {name: total / n_rows for name, total in policy_sums.items()}

    random_runs = randomized_metric_runs(
        primary_scores,
        candidate_ids,
        positive_item_ids,
        args.k,
        args.random_repeats,
        args.random_seed,
        args.chunk_size,
    )
    for metric_name in (f"hit@{args.k}", f"ndcg@{args.k}"):
        values = np.array([run[metric_name] for run in random_runs])
        metrics.setdefault("randomized", {})[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "repetitions": len(values),
        }

    analytic = analytic_means(primary_scores, positive_positions, args.k, args.chunk_size)
    metrics["analytic_uniform_tie_expectation"] = {
        f"hit@{args.k}": analytic["hit@k"],
        f"ndcg@{args.k}": analytic["ndcg@k"],
    }

    archived_collisions = 0
    hardened_collisions = 0
    for start in range(0, n_rows, args.chunk_size):
        end = min(start + args.chunk_size, n_rows)
        ids = candidate_ids[start:end]
        rows = user_ids[start:end]
        archived_collisions += count_key_collisions(
            archived_float32_keys(ids, rows, args.archived_seed), ids
        )
        hardened_collisions += count_key_collisions(
            hardened_uint64_keys(ids, rows, args.archived_seed), ids
        )

    report = {
        "schema_version": 1,
        "dataset": args.dataset,
        "input": {
            "path_name": args.input.name,
            "rows": n_rows,
            "candidates_per_row": n_candidates,
            "arrays": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": array_sha256(array),
                }
                for name, array in {
                    "candidate_ids": candidate_ids,
                    "primary_scores": primary_scores,
                    "positive_item_ids": positive_item_ids,
                    "user_ids": user_ids,
                }.items()
            },
        },
        "settings": {
            "k": args.k,
            "archived_seed": args.archived_seed,
            "random_repeats": args.random_repeats,
            "random_seed_start": args.random_seed,
            "permutation_seed": args.permutation_seed,
            "chunk_size": args.chunk_size,
        },
        "tie_diagnostics": tie_diagnostics(
            primary_scores,
            positive_positions=positive_positions,
            k=args.k,
        ),
        "secondary_key_collisions": {
            "archived_float32_excess_collisions": archived_collisions,
            "hardened_uint64_excess_collisions": hardened_collisions,
        },
        "metrics": metrics,
        "permutation_audit": {
            policy: full_permutation_audit(
                primary_scores,
                candidate_ids,
                user_ids,
                policy,
                args.archived_seed,
                args.permutation_seed,
                args.k,
                args.chunk_size,
            )
            for policy in ("stable", "archived_float32", "hardened_uint64")
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
