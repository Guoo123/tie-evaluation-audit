from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, repo_path
from .metrics import (
    analytic_single_positive_expectation,
    evaluate_policy,
    evaluate_policy_batched,
)
from .ranking import TiePolicy, rank_candidates


def _assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{label}: expected {expected:.12g} ± {tolerance:.3g}, got {actual:.12g}"
        )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify_distribution_manifest(root: Path) -> int:
    manifest = root / "ARTIFACT_MANIFEST.sha256"
    if not manifest.exists():
        raise AssertionError("ARTIFACT_MANIFEST.sha256 is missing")
    checked = 0
    errors: list[str] = []
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"manifest line {line_number} is malformed")
            continue
        path = root / relative
        if not path.exists():
            errors.append(f"manifest file is missing: {relative}")
            continue
        if _sha256(path) != expected:
            errors.append(f"manifest checksum mismatch: {relative}")
        checked += 1
    if errors:
        raise AssertionError("; ".join(errors))
    return checked


def _verify_run_integrity_manifest(path: Path, root: Path) -> int:
    if not path.exists():
        raise AssertionError(f"run integrity manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    checked = 0
    errors: list[str] = []
    for section in ("inputs", "outputs"):
        records = payload.get(section)
        if not isinstance(records, list):
            errors.append(f"{path.name}: {section} is not a list")
            continue
        for record in records:
            relative = record.get("path")
            expected = record.get("sha256")
            if not relative or not expected:
                errors.append(f"{path.name}: incomplete {section} record")
                continue
            target = root / relative
            if not target.exists():
                errors.append(f"{path.name}: listed file is missing: {relative}")
                continue
            if target.stat().st_size != int(record.get("bytes", -1)):
                errors.append(f"{path.name}: size mismatch: {relative}")
            if _sha256(target) != expected:
                errors.append(f"{path.name}: checksum mismatch: {relative}")
            checked += 1
    if errors:
        raise AssertionError("; ".join(errors))
    return checked


def smoke_checks() -> dict[str, Any]:
    users = np.array([7], dtype=np.int64)
    candidates = np.arange(31, dtype=np.int64)[None, :]
    positives = np.array([0], dtype=np.int64)
    tied_scores = np.zeros((1, 31), dtype=np.float32)

    stable = evaluate_policy(
        candidates,
        tied_scores,
        users,
        positives,
        policy=TiePolicy.STABLE_POSITIVE_FIRST,
        key_seed=20260316,
        k=10,
    )
    analytic = analytic_single_positive_expectation(tied_scores, candidates, positives, k=10)
    _assert_close(stable.hit_at_k, 1.0, 0.0, "all-tied stable Hit@10")
    _assert_close(stable.ndcg_at_k, 1.0, 0.0, "all-tied stable NDCG@10")
    _assert_close(analytic.hit_at_k, 10 / 31, 1e-12, "all-tied analytic Hit@10")
    expected_ndcg = sum(1 / math.log2(rank + 1) for rank in range(1, 11)) / 31
    _assert_close(analytic.ndcg_at_k, expected_ndcg, 1e-12, "all-tied analytic NDCG@10")

    rng = np.random.default_rng(1234)
    matrix = np.tile(np.arange(31, dtype=np.int64), (50, 1))
    scores = rng.integers(0, 4, size=matrix.shape).astype(np.float32)
    row_users = np.arange(50, dtype=np.int64)
    original = rank_candidates(
        matrix,
        scores,
        row_users,
        TiePolicy.HARDENED_UINT64,
        seed=20260316,
    )
    permuted_candidates = np.empty_like(matrix)
    permuted_scores = np.empty_like(scores)
    for row in range(len(matrix)):
        permutation = rng.permutation(matrix.shape[1])
        permuted_candidates[row] = matrix[row, permutation]
        permuted_scores[row] = scores[row, permutation]
    permuted = rank_candidates(
        permuted_candidates,
        permuted_scores,
        row_users,
        TiePolicy.HARDENED_UINT64,
        seed=20260316,
    )
    if not np.array_equal(original, permuted):
        raise AssertionError("hardened_uint64 ranking changed after row permutation")

    unique_scores = np.tile(np.arange(31, 0, -1, dtype=np.float32), (50, 1))
    stable_unique = rank_candidates(
        matrix,
        unique_scores,
        row_users,
        TiePolicy.STABLE_POSITIVE_FIRST,
        seed=20260316,
    )
    hardened_unique = rank_candidates(
        matrix,
        unique_scores,
        row_users,
        TiePolicy.HARDENED_UINT64,
        seed=20260316,
    )
    if not np.array_equal(stable_unique, hardened_unique):
        raise AssertionError("tie policy changed a strict-score ranking")

    return {
        "status": "pass",
        "all_tied_stable_ndcg_at_10": stable.ndcg_at_k,
        "all_tied_analytic_hit_at_10": analytic.hit_at_k,
        "all_tied_analytic_ndcg_at_10": analytic.ndcg_at_k,
        "hardened_permutation_rows": 50,
        "no_tie_equivalence_rows": 50,
    }


def _archive_row(table: pd.DataFrame, panel: str, dataset: str, score: str) -> pd.Series:
    rows = table[
        (table["panel"] == panel)
        & (table["dataset"] == dataset)
        & (table["score"] == score)
    ]
    if len(rows) != 1:
        raise AssertionError(
            f"expected one archived row for panel={panel}, dataset={dataset}, score={score}"
        )
    return rows.iloc[0]


def archived_checks(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config["_repo_root"])
    table_path = root / "results/archived/paper_table2.csv"
    table = pd.read_csv(table_path)
    if len(table) != 8:
        raise AssertionError(f"paper_table2.csv has {len(table)} rows, expected 8")
    required = [
        root / "results/archived/PROVENANCE.json",
        root / "results/archived/CLAIM_EVIDENCE.json",
        root / "results/archived/bpc/dataset_summary.json",
        root / "results/archived/bpc/original_resolved_config.json",
        root / "results/archived/bpc/original_run_manifest.json",
        root / "results/archived/bpc/tie_audit_summary.csv",
        root / "results/archived/movielens/ranking_results_by_tie_mode.csv",
        root / "results/archived/movielens/tie_audit_summary.csv",
        root / "results/archived/movielens/tie_audit_summary.json",
        root / "results/archived/movielens/experiment_summary.json",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.exists()]
    if missing:
        raise AssertionError(f"missing archived evidence: {missing}")

    provenance = json.loads((root / "results/archived/PROVENANCE.json").read_text())
    for dataset in ("bpc", "movielens"):
        absent = provenance[dataset]["not_available_in_supplied_zip"]
        if "historical frozen candidate rows" not in absent:
            raise AssertionError(f"{dataset} provenance does not disclose missing historical rows")

    claim_evidence = json.loads(
        (root / "results/archived/CLAIM_EVIDENCE.json").read_text(encoding="utf-8")
    )
    claims = claim_evidence.get("claims", [])
    claim_ids = {claim.get("id") for claim in claims}
    required_claim_ids = {
        "bpc_dataset_counts",
        "bpc_raw_ndcg",
        "bpc_raw_hit",
        "bpc_raw_tie_diagnostics",
        "bpc_centered_ndcg",
        "bpc_low_tie_control",
        "movielens_dataset_counts",
        "movielens_raw_ndcg_hit",
        "movielens_centered_ndcg",
        "movielens_popularity_control",
        "all_tied_theory",
    }
    if not required_claim_ids.issubset(claim_ids):
        missing_claims = sorted(required_claim_ids - claim_ids)
        raise AssertionError(f"claim-evidence map is missing: {missing_claims}")
    if claim_evidence.get("evidence_boundary", {}).get("historical_row_level_replay") is not False:
        raise AssertionError("claim-evidence map overstates historical row-level replay")

    bpc_summary = json.loads((root / "results/archived/bpc/dataset_summary.json").read_text())
    expected_bpc = config["bpc"]["expected_dataset"]
    mapping = {
        "n_users": "n_users",
        "n_items": "n_items",
        "n_train": "n_train",
        "n_val": "n_validation",
        "n_test": "n_test",
    }
    for archived_name, config_name in mapping.items():
        if int(bpc_summary[archived_name]) != int(expected_bpc[config_name]):
            raise AssertionError(f"BPC archived dataset count mismatch for {archived_name}")

    bpc = pd.read_csv(root / "results/archived/bpc/tie_audit_summary.csv")
    bpc_label_map = {
        "raw_count": "RawCount",
        "centered_count": "Centered count",
        "residualized_group_control": "Residualized (low tie)",
    }
    for score, paper_label in bpc_label_map.items():
        source = bpc[bpc["score"] == score]
        if len(source) != 1:
            raise AssertionError(f"missing BPC archived score {score}")
        paper = _archive_row(table, "NDCG@10", "BPC", paper_label)
        _assert_close(float(paper["stable"]), float(source.iloc[0]["stable_ndcg_at_10"]), 1e-12, f"BPC {score} stable")
        _assert_close(float(paper["keyed"]), float(source.iloc[0]["archived_keyed_ndcg_at_10"]), 1e-12, f"BPC {score} keyed")
    bpc_raw = bpc[bpc["score"] == "raw_count"].iloc[0]
    bpc_hit = _archive_row(table, "Hit@10", "BPC", "RawCount")
    _assert_close(float(bpc_hit["stable"]), float(bpc_raw["stable_hit_at_10"]), 1e-12, "BPC raw stable Hit")
    _assert_close(float(bpc_hit["keyed"]), float(bpc_raw["archived_keyed_hit_at_10"]), 1e-12, "BPC raw keyed Hit")

    ml = pd.read_csv(root / "results/archived/movielens/tie_audit_summary.csv")
    ml_label_map = {
        "raw_count": "RawCount",
        "centered_count": "Centered count",
        "pop_count": "Popularity",
    }
    for method, paper_label in ml_label_map.items():
        source = ml[ml["method"] == method]
        if len(source) != 1:
            raise AssertionError(f"missing MovieLens archived score {method}")
        paper = _archive_row(table, "NDCG@10", "MovieLens", paper_label)
        _assert_close(float(paper["stable"]), float(source.iloc[0]["stable_ndcg@10"]), 5e-5, f"MovieLens {method} stable")
        _assert_close(float(paper["keyed"]), float(source.iloc[0]["hash_ndcg@10"]), 5e-5, f"MovieLens {method} keyed")
    ml_raw = ml[ml["method"] == "raw_count"].iloc[0]
    ml_hit = _archive_row(table, "Hit@10", "MovieLens", "RawCount")
    _assert_close(float(ml_hit["stable"]), float(ml_raw["stable_hit@10"]), 1e-12, "MovieLens raw stable Hit")
    _assert_close(float(ml_hit["keyed"]), float(ml_raw["hash_hit@10"]), 1e-12, "MovieLens raw keyed Hit")

    calculated_change = table["keyed"].to_numpy(dtype=float) - table["stable"].to_numpy(dtype=float)
    if not np.allclose(calculated_change, table["change"].to_numpy(dtype=float), atol=5e-5, rtol=0):
        raise AssertionError("one or more archived Table 2 change values are inconsistent")

    manifest_files = _verify_distribution_manifest(root)
    return {
        "status": "pass",
        "paper_table_rows": len(table),
        "archived_files_checked": len(required),
        "manifest_files_checked": manifest_files,
        "claims_mapped": len(claims),
    }


def _row(metrics: pd.DataFrame, score: str, policy: str) -> pd.Series:
    subset = metrics[(metrics["score"] == score) & (metrics["policy"] == policy)]
    if len(subset) != 1:
        raise AssertionError(
            f"expected one result for score={score}, policy={policy}; got {len(subset)}"
        )
    return subset.iloc[0]


def _frozen_rows_checks(
    output_dir: Path,
    metrics: pd.DataFrame,
    score_names: list[str],
    *,
    key_seed: int,
    k: int,
    expected_rows: int,
    expected_width: int,
    full_replay: bool,
) -> dict[str, Any]:
    rows_path = output_dir / "frozen_candidate_rows.npz"
    if not rows_path.exists():
        raise AssertionError(f"missing frozen rows: {rows_path}")
    with np.load(rows_path) as frozen:
        users = frozen["user_indices"].astype(np.int64, copy=False)
        positives = frozen["positive_items"].astype(np.int64, copy=False)
        candidates = frozen["candidates"].astype(np.int64, copy=False)
    if candidates.shape != (expected_rows, expected_width):
        raise AssertionError(
            f"frozen candidate shape {candidates.shape} != {(expected_rows, expected_width)}"
        )
    if not np.array_equal(candidates[:, 0], positives):
        raise AssertionError("the held-out positive is not stored in column zero")
    occurrences = (candidates == positives[:, None]).sum(axis=1)
    if not np.all(occurrences == 1):
        raise AssertionError("one or more frozen rows do not contain the positive exactly once")

    rng = np.random.default_rng(7717)
    subset_size = min(512, expected_rows)
    subset_indices = np.sort(rng.choice(expected_rows, size=subset_size, replace=False))
    permutation_checks = 0
    replayed_metric_rows = 0
    for score_name in score_names:
        score_path = output_dir / f"candidate_scores_{score_name}.npy"
        if not score_path.exists():
            raise AssertionError(f"missing retained candidate scores: {score_path.name}")
        scores = np.load(score_path, mmap_mode="r")
        if scores.shape != candidates.shape:
            raise AssertionError(f"score shape for {score_name} does not match frozen rows")

        subset_candidates = candidates[subset_indices]
        subset_scores = np.asarray(scores[subset_indices])
        subset_users = users[subset_indices]
        original = rank_candidates(
            subset_candidates,
            subset_scores,
            subset_users,
            TiePolicy.HARDENED_UINT64,
            seed=key_seed,
        )
        permuted_candidates = np.empty_like(subset_candidates)
        permuted_scores = np.empty_like(subset_scores)
        for row in range(subset_size):
            permutation = rng.permutation(expected_width)
            permuted_candidates[row] = subset_candidates[row, permutation]
            permuted_scores[row] = subset_scores[row, permutation]
        permuted = rank_candidates(
            permuted_candidates,
            permuted_scores,
            subset_users,
            TiePolicy.HARDENED_UINT64,
            seed=key_seed,
        )
        if not np.array_equal(original, permuted):
            raise AssertionError(f"hardened permutation failure in regenerated {score_name}")
        permutation_checks += subset_size

        if full_replay:
            for policy in (
                TiePolicy.STABLE_POSITIVE_FIRST,
                TiePolicy.ARCHIVED_FLOAT32,
                TiePolicy.HARDENED_UINT64,
            ):
                replayed = evaluate_policy_batched(
                    candidates,
                    scores,
                    users,
                    positives,
                    policy=policy,
                    key_seed=key_seed,
                    k=k,
                    batch_rows=100_000,
                )
                recorded = _row(metrics, score_name, policy.value)
                for field in ("hit_at_k", "ndcg_at_k", "mrr"):
                    _assert_close(
                        float(getattr(replayed, field)),
                        float(recorded[field]),
                        1e-12,
                        f"frozen replay {score_name} {policy.value} {field}",
                    )
                replayed_metric_rows += 1
            analytic = analytic_single_positive_expectation(
                scores,
                candidates,
                positives,
                k=k,
            )
            recorded = _row(metrics, score_name, "analytic_uniform_tie_expectation")
            for field in ("hit_at_k", "ndcg_at_k", "mrr"):
                _assert_close(
                    float(getattr(analytic, field)),
                    float(recorded[field]),
                    1e-12,
                    f"frozen replay {score_name} analytic {field}",
                )
            replayed_metric_rows += 1

    return {
        "frozen_rows": expected_rows,
        "candidate_width": expected_width,
        "score_matrices_checked": len(score_names),
        "hardened_permutation_row_checks": permutation_checks,
        "metric_rows_replayed": replayed_metric_rows,
    }


def _compare_expected(
    metrics: pd.DataFrame,
    expected: dict[str, Any],
    score: str,
    tolerance: float,
    fields: tuple[str, ...] = (
        "stable_ndcg_at_10",
        "archived_keyed_ndcg_at_10",
        "stable_hit_at_10",
        "archived_keyed_hit_at_10",
    ),
) -> list[dict[str, Any]]:
    stable = _row(metrics, score, TiePolicy.STABLE_POSITIVE_FIRST.value)
    keyed = _row(metrics, score, TiePolicy.ARCHIVED_FLOAT32.value)
    values = {
        "stable_ndcg_at_10": stable["ndcg_at_k"],
        "archived_keyed_ndcg_at_10": keyed["ndcg_at_k"],
        "stable_hit_at_10": stable["hit_at_k"],
        "archived_keyed_hit_at_10": keyed["hit_at_k"],
    }
    comparisons: list[dict[str, Any]] = []
    for field in fields:
        if field not in expected:
            continue
        actual = float(values[field])
        expected_value = float(expected[field])
        _assert_close(actual, expected_value, tolerance, f"{score} {field}")
        comparisons.append(
            {
                "score": score,
                "quantity": field,
                "actual": actual,
                "expected": expected_value,
                "absolute_difference": abs(actual - expected_value),
            }
        )
    return comparisons


def regenerated_checks(config: dict[str, Any]) -> dict[str, Any]:
    output_root = repo_path(config, config["artifact"]["output_root"])
    report: dict[str, Any] = {"status": "not-run", "datasets": {}}
    k = int(config["artifact"]["cutoff"])
    width = int(config["artifact"]["candidate_count"])

    ml_dir = output_root / "movielens"
    ml_path = ml_dir / "metrics_by_policy.csv"
    ml_summary_path = ml_dir / "run_summary.json"
    if ml_path.exists() and ml_summary_path.exists():
        metrics = pd.read_csv(ml_path)
        summary = json.loads(ml_summary_path.read_text())
        tolerance = float(config["movielens"]["tolerance"]["metric_absolute"])
        comparisons: list[dict[str, Any]] = []
        for score in ("raw_count", "centered_count", "popularity"):
            comparisons.extend(
                _compare_expected(
                    metrics,
                    config["movielens"]["expected_archived"][score],
                    score,
                    tolerance,
                )
            )
        expected_dataset = config["movielens"]["expected_dataset"]
        for field in ("n_users", "n_items", "n_train", "n_test"):
            actual = int(summary["dataset"][field])
            if actual != int(expected_dataset[field]):
                raise AssertionError(
                    f"MovieLens dataset count {field}: {actual} != {expected_dataset[field]}"
                )
        frozen = _frozen_rows_checks(
            ml_dir,
            metrics,
            ["raw_count", "centered_count", "popularity"],
            key_seed=int(config["policies"]["movielens_key_seed"]),
            k=k,
            expected_rows=int(expected_dataset["n_test"]),
            expected_width=width,
            full_replay=True,
        )
        for required_name in (
            "runtime_manifest.json",
            "resolved_config.json",
            "run_integrity_manifest.json",
            "diagnostics.json",
        ):
            if not (ml_dir / required_name).exists():
                raise AssertionError(f"MovieLens output is missing {required_name}")
        integrity_files = _verify_run_integrity_manifest(
            ml_dir / "run_integrity_manifest.json", Path(config["_repo_root"])
        )
        report["datasets"]["movielens"] = {
            "status": "pass",
            "comparisons": comparisons,
            "frozen_evidence": frozen,
            "integrity_files_checked": integrity_files,
        }
        report["status"] = "pass"

    bpc_dir = output_root / "bpc"
    bpc_path = bpc_dir / "metrics_by_policy.csv"
    bpc_summary_path = bpc_dir / "run_summary.json"
    if bpc_path.exists() and bpc_summary_path.exists():
        metrics = pd.read_csv(bpc_path)
        summary = json.loads(bpc_summary_path.read_text())
        if summary.get("paper_scale"):
            tolerance = float(config["bpc"]["tolerance"]["metric_absolute"])
            comparisons: list[dict[str, Any]] = []
            expected_scores = config["bpc"]["expected_archived"]
            available_scores = set(metrics["score"])
            required_scores = ["raw_count", "centered_count"]
            for score in required_scores:
                comparisons.extend(
                    _compare_expected(metrics, expected_scores[score], score, tolerance)
                )
            low_tie_present = "residualized_group_control" in available_scores
            if low_tie_present:
                comparisons.extend(
                    _compare_expected(
                        metrics,
                        expected_scores["residualized_group_control"],
                        "residualized_group_control",
                        tolerance,
                        fields=("stable_ndcg_at_10", "archived_keyed_ndcg_at_10"),
                    )
                )
            expected_dataset = config["bpc"]["expected_dataset"]
            for field in ("n_users", "n_items", "n_train", "n_validation", "n_test"):
                actual = int(summary["dataset"][field])
                if actual != int(expected_dataset[field]):
                    raise AssertionError(
                        f"BPC dataset count {field}: {actual} != {expected_dataset[field]}"
                    )
            score_files = ["raw_count", "centered_count", "item_residual_only"]
            if low_tie_present:
                score_files.append("residualized_group_control")
            frozen = _frozen_rows_checks(
                bpc_dir,
                metrics,
                score_files,
                key_seed=int(config["policies"]["bpc_key_seed"]),
                k=k,
                expected_rows=int(expected_dataset["n_test"]),
                expected_width=width,
                full_replay=False,
            )
            for required_name in (
                "runtime_manifest.json",
                "resolved_config.json",
                "run_integrity_manifest.json",
                "diagnostics.json",
            ):
                if not (bpc_dir / required_name).exists():
                    raise AssertionError(f"BPC output is missing {required_name}")
            integrity_files = _verify_run_integrity_manifest(
                bpc_dir / "run_integrity_manifest.json", Path(config["_repo_root"])
            )
            report["datasets"]["bpc"] = {
                "status": "pass" if low_tie_present else "partial-paper-scale-no-low-tie-control",
                "comparisons": comparisons,
                "frozen_evidence": frozen,
                "integrity_files_checked": integrity_files,
            }
            report["status"] = "pass"
        else:
            report["datasets"]["bpc"] = {
                "status": "subset-not-compared-to-paper",
                "evaluated_rows": summary.get("evaluated_rows"),
            }
    if report["status"] == "not-run":
        report["reason"] = "No completed full-scale regenerated metrics and run summary were found."
    return report


def run_verification(config: dict[str, Any], level: str) -> dict[str, Any]:
    report: dict[str, Any] = {"level": level, "smoke": smoke_checks()}
    if level in {"archive", "full"}:
        report["archive"] = archived_checks(config)
    if level == "full":
        regenerated = regenerated_checks(config)
        if regenerated.get("status") != "pass":
            raise AssertionError(
                "full verification requires at least one completed full-scale regenerated "
                "dataset; run MovieLens or BPC first"
            )
        report["regenerated"] = regenerated
    report["status"] = "pass"
    return report


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Verify the tie-evaluation artifact.")
    parser.add_argument("--config", default=str(repo_root / "configs/paper.yaml"))
    parser.add_argument("--level", choices=["smoke", "archive", "full"], default="smoke")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional path for the JSON verification report. When omitted, the report "
            "is printed without modifying the checkout."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    report = run_verification(config, args.level)
    if args.output is not None:
        destination = Path(args.output).expanduser()
        if not destination.is_absolute():
            destination = repo_root / destination
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
