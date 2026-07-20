from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tie_eval.config import load_config
from tie_eval.verify import archived_checks


ROOT = Path(__file__).resolve().parents[1]


def test_archived_bundle_is_internally_consistent() -> None:
    config = load_config(ROOT / "configs/paper.yaml")
    result = archived_checks(config)
    assert result["status"] == "pass"


def test_provenance_does_not_claim_historical_row_replay() -> None:
    provenance = json.loads((ROOT / "results/archived/PROVENANCE.json").read_text())
    assert "historical frozen candidate rows" in provenance["bpc"]["not_available_in_supplied_zip"]
    assert "historical frozen candidate rows" in provenance["movielens"]["not_available_in_supplied_zip"]


def test_paper_table_contains_the_eight_submitted_rows() -> None:
    table = pd.read_csv(ROOT / "results/archived/paper_table2.csv")
    assert len(table) == 8
    raw_bpc = table[(table["dataset"] == "BPC") & (table["score"] == "RawCount")]
    assert set(raw_bpc["panel"]) == {"NDCG@10", "Hit@10"}
