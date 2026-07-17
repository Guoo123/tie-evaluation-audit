#!/usr/bin/env python3
"""Verify manuscript-table arithmetic and local snapshot comparisons."""

from __future__ import annotations

import csv
import json
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


def verify_manuscript_table() -> int:
    rows = read_csv("manuscript_table2.csv")
    require(len(rows) == 8, f"expected 8 manuscript Table 2 rows, found {len(rows)}")

    for row in rows:
        stable = Decimal(row["stable"])
        keyed = Decimal(row["keyed"])
        reported_change = Decimal(row["change"])
        require(Decimal(0) <= stable <= Decimal(1), f"stable value out of range: {row}")
        require(Decimal(0) <= keyed <= Decimal(1), f"keyed value out of range: {row}")
        require(reported_change <= 0, f"change must use keyed-minus-stable sign: {row}")

        arithmetic_change = keyed - stable
        if row["score"] == "Popularity" and row["dataset"] == "MovieLens":
            # Stable and keyed are shown to four decimals, while the delta is
            # retained at 2.2e-5. The rounded endpoints cannot recover it.
            require(
                abs(reported_change) < Decimal("0.0001"),
                "MovieLens popularity full-precision delta exceeds endpoint rounding interval",
            )
        else:
            require(
                abs(arithmetic_change - reported_change) <= Decimal("0.0001"),
                f"reported change is inconsistent with rounded endpoints: {row}",
            )
    return len(rows)


def verify_diagnostics() -> int:
    rows = read_csv("manuscript_bpc_tie_diagnostics.csv")
    require(len(rows) == 4, f"expected 4 BPC diagnostics, found {len(rows)}")
    for row in rows:
        value = Decimal(row["value_fraction"])
        require(Decimal(0) <= value <= Decimal(1), f"diagnostic fraction out of range: {row}")
    return len(rows)


def verify_movielens_comparison() -> int:
    rows = read_csv("movielens_manuscript_vs_local.csv")
    require(len(rows) == 4, f"expected 4 MovieLens comparison rows, found {len(rows)}")
    for row in rows:
        manuscript = Decimal(row["manuscript_keyed"])
        local = Decimal(row["local_keyed"])
        stored = Decimal(row["local_minus_manuscript"])
        require(
            abs((local - manuscript) - stored) <= Decimal("1e-15"),
            f"MovieLens comparison arithmetic mismatch: {row}",
        )
        require(stored != 0, "local MovieLens snapshot must not be presented as the historical run")
    return len(rows)


def verify_bpc_comparison() -> int:
    rows = read_csv("bpc_manuscript_vs_local.csv")
    require(len(rows) == 2, f"expected 2 BPC comparison rows, found {len(rows)}")
    for row in rows:
        manuscript_stable = Decimal(row["manuscript_stable"])
        manuscript_keyed = Decimal(row["manuscript_keyed"])
        manuscript_change = Decimal(row["manuscript_change_keyed_minus_stable"])
        local_stable = Decimal(row["local_stable"])
        local_hash = Decimal(row["local_hash"])
        local_unsigned_delta = Decimal(row["local_delta_stable_minus_hash"])
        require(manuscript_stable == local_stable, f"BPC stable endpoints disagree: {row}")
        require(manuscript_keyed == local_hash, f"BPC keyed endpoints disagree: {row}")
        require(
            abs((local_stable - local_hash) - local_unsigned_delta) <= Decimal("0.0001"),
            f"BPC local delta is inconsistent with displayed endpoints: {row}",
        )
        require(
            abs((-local_unsigned_delta) - manuscript_change) <= Decimal("0.0001"),
            f"BPC signed change exceeds the documented precision difference: {row}",
        )
    return len(rows)


def verify_source_manifest() -> int:
    rows = read_csv("source_manifest.csv")
    for row in rows:
        path = ROOT / row["file"]
        require(path.is_file(), f"source manifest references missing file: {row['file']}")
    return len(rows)


def verify_configs() -> int:
    count = 0
    for path in sorted((ROOT / "configs").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        require(payload.get("schema_version") == 1, f"unexpected config schema: {path.name}")
        count += 1
    require(count == 2, f"expected two canonical JSON configs, found {count}")
    return count


def main() -> None:
    counts = {
        "manuscript_table_rows": verify_manuscript_table(),
        "bpc_diagnostic_rows": verify_diagnostics(),
        "movielens_comparison_rows": verify_movielens_comparison(),
        "bpc_comparison_rows": verify_bpc_comparison(),
        "source_manifest_rows": verify_source_manifest(),
        "canonical_configs": verify_configs(),
    }
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(
        "PASS: stored aggregate arithmetic and declared file/config structure are "
        "internally consistent; this is not a row-level reproduction."
    )


if __name__ == "__main__":
    main()
