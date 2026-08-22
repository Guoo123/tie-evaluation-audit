from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_camera_ready_config_is_separate_and_uses_100_seeds() -> None:
    accepted = yaml.safe_load((ROOT / "configs/paper.yaml").read_text(encoding="utf-8"))
    camera = yaml.safe_load((ROOT / "configs/camera_ready.yaml").read_text(encoding="utf-8"))
    assert accepted["policies"]["randomized_repetitions"] == 20
    assert camera["policies"]["randomized_repetitions"] == 100
    assert camera["bpc"]["expected_archived"]["residualized_group_control"]["any_tie_rate"] == 0.0039


def test_amazon_reconstruction_and_tie_counts_match_camera_ready_claims() -> None:
    root = ROOT / "results/camera_ready/amazon"
    reproduction = pd.read_csv(root / "reproduction_check.csv")
    assert len(reproduction) == 84
    assert set(reproduction["status"]) == {"PASS"}
    assert float(reproduction["absolute_difference"].max()) == 0.0

    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    raw = diagnostics["scores"]["raw_count"]
    residualized = diagnostics["scores"]["residualized_group_control"]
    assert diagnostics["rows"] == 30_000
    assert diagnostics["candidates_per_row"] == 31
    assert raw["any_exact_tie_rows"] == 30_000
    assert raw["topk_boundary_tie_rows"] == 29_981
    assert raw["positive_in_tie_rows"] == 29_651
    assert residualized["any_exact_tie_rows"] == 117


def test_camera_ready_policy_values_and_float32_audit() -> None:
    root = ROOT / "results/camera_ready/amazon"
    policies = pd.read_csv(root / "policy_comparison_for_paper.csv")
    row = policies[
        (policies["score"] == "raw_count") & (policies["metric"] == "NDCG@10")
    ].iloc[0]
    assert row["stable_positive_first"] == pytest.approx(0.847419307899178)
    assert row["hardened_uint64"] == pytest.approx(0.17023696678473144)
    assert row["analytic_uniform_tie_expectation"] == pytest.approx(0.16930035129969778)
    assert row["randomized_mean_100"] == pytest.approx(0.16942719461148026)

    collision = json.loads(
        (root / "float32_collision_summary.json").read_text(encoding="utf-8")
    )
    assert collision["distinct_item_identities_sharing_float32_secondary_key_rows"] == 1
    for score in collision["per_score"].values():
        assert score["hardened_uint64_row_order_dependent_rows"] == 0
        assert score["top10_membership_difference_rows"] == 0
        assert score["positive_rank_difference_rows"] == 0


def test_paper_output_script_writes_tables_without_a_figure(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_paper_outputs.py"),
            "--config",
            str(ROOT / "configs/camera_ready.yaml"),
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        cwd=ROOT,
    )
    table2 = pd.read_csv(tmp_path / "table2_cross_domain.csv")
    table3 = pd.read_csv(tmp_path / "table3_amazon_policies.csv")
    assert len(table2) == 6
    assert len(table3) == 3
    assert not (tmp_path / "figure1.png").exists()
    assert (tmp_path / "manifest.json").exists()
