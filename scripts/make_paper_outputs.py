#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.config import load_config, repo_path
from tie_eval.provenance import file_record, write_json
from tie_eval.ranking import TiePolicy

CAMERA_READY_EVIDENCE = (
    REPO_ROOT / "results/camera_ready/amazon/policy_comparison_for_paper.csv"
)

CROSS_DOMAIN_ROWS = [
    ("Amazon Beauty", "raw_count", "Weighted attribute overlap", "NDCG@10", "ndcg_at_k"),
    ("Amazon Beauty", "centered_count", "Centered attribute overlap", "NDCG@10", "ndcg_at_k"),
    (
        "Amazon Beauty",
        "residualized_group_control",
        "Residualized attribute score",
        "NDCG@10",
        "ndcg_at_k",
    ),
    ("MovieLens", "raw_count", "Tag-attribute overlap", "NDCG@10", "ndcg_at_k"),
    ("MovieLens", "centered_count", "Centered tag overlap", "NDCG@10", "ndcg_at_k"),
    ("MovieLens", "popularity", "Item popularity", "NDCG@10", "ndcg_at_k"),
]

ARCHIVED_LABELS = {
    ("Amazon Beauty", "raw_count"): ("BPC", "RawCount"),
    ("Amazon Beauty", "centered_count"): ("BPC", "Centered count"),
    ("Amazon Beauty", "residualized_group_control"): ("BPC", "Residualized (low tie)"),
    ("MovieLens", "raw_count"): ("MovieLens", "RawCount"),
    ("MovieLens", "centered_count"): ("MovieLens", "Centered count"),
    ("MovieLens", "popularity"): ("MovieLens", "Popularity"),
}


def _load_regenerated(config: dict[str, Any], dataset: str) -> pd.DataFrame | None:
    root = repo_path(config, config["artifact"]["output_root"]) / dataset
    metrics_path = root / "metrics_by_policy.csv"
    summary_path = root / "run_summary.json"
    if not metrics_path.exists() or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset == "bpc" and not bool(summary.get("paper_scale")):
        return None
    return pd.read_csv(metrics_path)


def _metric_value(frame: pd.DataFrame, score: str, policy: str, metric: str) -> float:
    row = frame[(frame["score"] == score) & (frame["policy"] == policy)]
    if len(row) != 1:
        raise ValueError(f"Missing unique row for score={score}, policy={policy}")
    return float(row.iloc[0][metric])


def build_cross_domain_table(
    config: dict[str, Any], evidence_path: Path
) -> tuple[pd.DataFrame, list[Path]]:
    archived_path = REPO_ROOT / "results/archived/paper_table2.csv"
    archived = pd.read_csv(archived_path)
    camera_ready = pd.read_csv(evidence_path)
    regenerated = {
        "Amazon Beauty": _load_regenerated(config, "bpc"),
        "MovieLens": _load_regenerated(config, "movielens"),
    }
    dataset_keys = {"Amazon Beauty": "bpc", "MovieLens": "movielens"}
    stable = TiePolicy.STABLE_POSITIVE_FIRST.value
    hardened = TiePolicy.HARDENED_UINT64.value
    historical = TiePolicy.ARCHIVED_FLOAT32.value

    rows: list[dict[str, Any]] = []
    input_paths = [archived_path, evidence_path]
    for dataset, score, label, panel, metric in CROSS_DOMAIN_ROWS:
        frame = regenerated[dataset]
        source = "accepted aggregate"
        hash_policy = historical
        change_value: float | None = None

        if dataset == "Amazon Beauty":
            camera_row = camera_ready[
                (camera_ready["dataset"] == "amazon")
                & (camera_ready["score"] == score)
                & (camera_ready["metric"] == panel)
            ]
            if len(camera_row) == 1:
                input_order = float(camera_row.iloc[0]["stable_positive_first"])
                hash_value = float(camera_row.iloc[0]["hardened_uint64"])
                change_value = hash_value - input_order
                source = "camera-ready deterministic reconstruction"
                hash_policy = hardened
                frame = None
                use_archived = False
            else:
                use_archived = True
        else:
            use_archived = True

        if dataset != "Amazon Beauty" and frame is not None:
            available = set(frame.loc[frame["score"] == score, "policy"])
            if stable in available and hardened in available:
                input_order = _metric_value(frame, score, stable, metric)
                hash_value = _metric_value(frame, score, hardened, metric)
                source = "regenerated hardened_uint64"
                hash_policy = hardened
                use_archived = False
            elif stable in available and historical in available:
                input_order = _metric_value(frame, score, stable, metric)
                hash_value = _metric_value(frame, score, historical, metric)
                source = "regenerated archived_float32"
                use_archived = False
            else:
                frame = None
        if use_archived:
            archived_dataset, archived_score = ARCHIVED_LABELS[(dataset, score)]
            archived_row = archived[
                (archived["panel"] == panel)
                & (archived["dataset"] == archived_dataset)
                & (archived["score"] == archived_score)
            ]
            if len(archived_row) != 1:
                raise ValueError(
                    f"No accepted aggregate row for dataset={dataset}, score={score}, panel={panel}"
                )
            input_order = float(archived_row.iloc[0]["stable"])
            hash_value = float(archived_row.iloc[0]["keyed"])
            change_value = float(archived_row.iloc[0]["change"])

        if change_value is None:
            change_value = hash_value - input_order

        rows.append(
            {
                "dataset": dataset,
                "score": label,
                "metric": panel,
                "input_order": input_order,
                "hash_tie_break": hash_value,
                "change": change_value,
                "hash_policy": hash_policy,
                "source": source,
            }
        )

    for dataset in dataset_keys.values():
        path = repo_path(config, config["artifact"]["output_root"]) / dataset / "metrics_by_policy.csv"
        if path.exists():
            input_paths.append(path)
    return pd.DataFrame(rows), input_paths


def build_amazon_policy_table(evidence_path: Path) -> pd.DataFrame:
    evidence = pd.read_csv(evidence_path)
    selected = evidence[
        (evidence["dataset"] == "amazon")
        & (evidence["metric"] == "NDCG@10")
        & evidence["score"].isin(
            ["raw_count", "centered_count", "residualized_group_control"]
        )
    ].copy()
    labels = {
        "raw_count": "Weighted attribute overlap",
        "centered_count": "Centered attribute overlap",
        "residualized_group_control": "Residualized attribute score",
    }
    order = {score: position for position, score in enumerate(labels)}
    selected["_order"] = selected["score"].map(order)
    selected = selected.sort_values("_order")
    selected["score"] = selected["score"].map(labels)
    selected["randomized_mean_plus_minus_sd"] = selected.apply(
        lambda row: f"{row['randomized_mean_100']:.6f} ± {row['randomized_sd_100']:.6f}",
        axis=1,
    )
    return selected[
        [
            "score",
            "stable_positive_first",
            "hardened_uint64",
            "analytic_uniform_tie_expectation",
            "randomized_mean_100",
            "randomized_sd_100",
            "randomized_mean_plus_minus_sd",
        ]
    ].rename(
        columns={
            "stable_positive_first": "input_order",
            "hardened_uint64": "hash_tie_break",
            "analytic_uniform_tie_expectation": "expected_over_ties",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the camera-ready cross-domain and Amazon policy tables."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/camera_ready.yaml"))
    parser.add_argument(
        "--camera-ready-evidence",
        default=str(CAMERA_READY_EVIDENCE),
        help="Amazon policy-comparison CSV from the camera-ready evaluator audit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory; defaults to results/regenerated/paper_outputs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evidence_path = Path(args.camera_ready_evidence).resolve()
    if not evidence_path.is_file():
        raise FileNotFoundError(f"camera-ready evidence not found: {evidence_path}")
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else repo_path(config, config["artifact"]["output_root"]) / "paper_outputs"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cross_domain, cross_inputs = build_cross_domain_table(config, evidence_path)
    amazon_policies = build_amazon_policy_table(evidence_path)
    table2_path = output_dir / "table2_cross_domain.csv"
    table3_path = output_dir / "table3_amazon_policies.csv"
    cross_domain.to_csv(table2_path, index=False)
    amazon_policies.to_csv(table3_path, index=False)

    manifest_path = output_dir / "manifest.json"
    write_json(
        {
            "schema_version": 2,
            "purpose": "FRAME 2026 camera-ready Tables 2 and 3 data products",
            "inputs": [
                file_record(path, root=REPO_ROOT)
                for path in cross_inputs
            ],
            "outputs": [
                file_record(table2_path, root=REPO_ROOT),
                file_record(table3_path, root=REPO_ROOT),
            ],
            "notes": [
                "No figure is generated because the redundant submission figure was removed.",
                "Accepted cross-domain aggregate values remain unchanged when no full-scale regenerated run is present.",
                "The Amazon policy table uses the committed camera-ready evaluator-only evidence.",
            ],
        },
        manifest_path,
    )
    print(f"Wrote {table2_path}, {table3_path}, and {manifest_path}.")


if __name__ == "__main__":
    main()
