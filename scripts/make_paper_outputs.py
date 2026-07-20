#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.config import load_config, repo_path
from tie_eval.provenance import file_record, write_json
from tie_eval.ranking import TiePolicy


def _load_regenerated(config: dict[str, Any], dataset: str) -> pd.DataFrame | None:
    root = repo_path(config, config["artifact"]["output_root"]) / dataset
    path = root / "metrics_by_policy.csv"
    summary_path = root / "run_summary.json"
    if not path.exists() or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dataset == "bpc" and not bool(summary.get("paper_scale")):
        return None
    return pd.read_csv(path)


def _value(frame: pd.DataFrame, score: str, policy: str, metric: str) -> float:
    row = frame[(frame["score"] == score) & (frame["policy"] == policy)]
    if len(row) != 1:
        raise ValueError(f"Missing unique row for score={score}, policy={policy}")
    return float(row.iloc[0][metric])


def build_table(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    archived = pd.read_csv(REPO_ROOT / "results/archived/paper_table2.csv")
    regenerated = {
        "BPC": _load_regenerated(config, "bpc"),
        "MovieLens": _load_regenerated(config, "movielens"),
    }
    stable = TiePolicy.STABLE_POSITIVE_FIRST.value
    keyed = TiePolicy.ARCHIVED_FLOAT32.value
    specifications = [
        ("NDCG@10", "BPC", "raw_count", "RawCount", "ndcg_at_k"),
        ("NDCG@10", "BPC", "centered_count", "Centered count", "ndcg_at_k"),
        (
            "NDCG@10",
            "BPC",
            "residualized_group_control",
            "Residualized (low tie)",
            "ndcg_at_k",
        ),
        ("NDCG@10", "MovieLens", "raw_count", "RawCount", "ndcg_at_k"),
        ("NDCG@10", "MovieLens", "centered_count", "Centered count", "ndcg_at_k"),
        ("NDCG@10", "MovieLens", "popularity", "Popularity", "ndcg_at_k"),
        ("Hit@10", "BPC", "raw_count", "RawCount", "hit_at_k"),
        ("Hit@10", "MovieLens", "raw_count", "RawCount", "hit_at_k"),
    ]

    rows: list[dict[str, Any]] = []
    source_counts = {"archived": 0, "regenerated": 0}
    for panel, dataset, score, label, metric in specifications:
        frame = regenerated[dataset]
        can_use_regenerated = (
            frame is not None
            and score in set(frame["score"])
            and stable in set(frame.loc[frame["score"] == score, "policy"])
            and keyed in set(frame.loc[frame["score"] == score, "policy"])
        )
        if can_use_regenerated:
            stable_value = _value(frame, score, stable, metric)
            keyed_value = _value(frame, score, keyed, metric)
            change_value = keyed_value - stable_value
            source = "regenerated"
        else:
            archived_row = archived[
                (archived["panel"] == panel)
                & (archived["dataset"] == dataset)
                & (archived["score"] == label)
            ]
            if len(archived_row) != 1:
                raise ValueError(
                    f"No regenerated or archived value for panel={panel}, dataset={dataset}, score={label}"
                )
            stable_value = float(archived_row.iloc[0]["stable"])
            keyed_value = float(archived_row.iloc[0]["keyed"])
            # Preserve the retained full-precision change. Some submitted cells display
            # stable and keyed values at four decimals while reporting a smaller
            # full-precision delta (the MovieLens popularity control).
            change_value = float(archived_row.iloc[0]["change"])
            source = "archived"
        source_counts[source] += 1
        rows.append(
            {
                "panel": panel,
                "dataset": dataset,
                "score": label,
                "stable": stable_value,
                "keyed": keyed_value,
                "change": change_value,
                "source": source,
            }
        )

    overall = (
        "regenerated"
        if source_counts["archived"] == 0
        else "archived"
        if source_counts["regenerated"] == 0
        else "mixed"
    )
    return pd.DataFrame(rows), {"overall": overall, "row_counts": source_counts}


def make_figure(table: pd.DataFrame, destination: Path) -> None:
    panel = table[table["panel"] == "NDCG@10"].copy()
    labels = [f"{dataset}\n{score}" for dataset, score in zip(panel["dataset"], panel["score"])]
    x = np.arange(len(panel))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.2))
    axis.bar(x - width / 2, panel["stable"], width, label="Stable input order")
    axis.bar(x + width / 2, panel["keyed"], width, label="Archived keyed order")
    axis.set_ylabel("NDCG@10")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_ylim(0, 0.95)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    for offset, column in ((-width / 2, "stable"), (width / 2, "keyed")):
        for position, value in zip(x, panel[column]):
            axis.text(
                position + offset,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate Table 2 and Figure 1 data products.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/paper.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    table, source = build_table(config)
    output_dir = repo_path(config, config["artifact"]["output_root"]) / "paper_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "table2.csv", index=False)
    make_figure(table, output_dir / "figure1.png")
    input_paths = [REPO_ROOT / "results/archived/paper_table2.csv"]
    for dataset in ("bpc", "movielens"):
        path = repo_path(config, config["artifact"]["output_root"]) / dataset / "metrics_by_policy.csv"
        if path.exists():
            input_paths.append(path)
    write_json(
        {
            "source": source,
            "inputs": [file_record(path, root=REPO_ROOT) for path in input_paths],
            "outputs": [
                file_record(output_dir / "table2.csv", root=REPO_ROOT),
                file_record(output_dir / "figure1.png", root=REPO_ROOT),
            ],
            "note": (
                "Each row states whether it came from a completed full-scale regeneration "
                "or from archived aggregate evidence. Keyed values use the archived float32 "
                "policy to match the submitted table; hardened and analytic values remain in "
                "metrics_by_policy.csv."
            ),
        },
        output_dir / "manifest.json",
    )
    print(
        f"Wrote {output_dir / 'table2.csv'} and {output_dir / 'figure1.png'} "
        f"(source={source['overall']})."
    )


if __name__ == "__main__":
    main()
