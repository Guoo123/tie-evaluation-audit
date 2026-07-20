#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.config import load_config, repo_path

PINNED = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.14.1",
    "scikit-learn": "1.5.2",
    "joblib": "1.4.2",
    "pyarrow": "18.1.0",
    "duckdb": "1.1.3",
    "PyYAML": "6.0.2",
    "requests": "2.32.3",
    "tqdm": "4.67.1",
    "orjson": "3.10.12",
    "matplotlib": "3.9.3",
}


def gib(value: int | float) -> float:
    return float(value) / (1024**3)


def physical_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        return page_size * pages
    except (AttributeError, OSError, ValueError):
        return None


def package_status() -> dict[str, dict[str, str | bool]]:
    result: dict[str, dict[str, str | bool]] = {}
    for package, expected in PINNED.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = "not-installed"
        result[package] = {
            "expected": expected,
            "actual": actual,
            "matches": actual == expected,
        }
    return result


def dataset_status(config: dict[str, Any], datasets: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if "movielens" in datasets:
        root = repo_path(config, config["movielens"]["download"]["extract_dir"])
        required = [root / "ratings.csv", root / "genome-tags.csv", root / "genome-scores.csv"]
        data["movielens"] = {
            "ready": all(path.exists() for path in required),
            "required_files": {
                str(path.relative_to(REPO_ROOT)): path.exists() for path in required
            },
            "recommended_free_disk_gib": 10,
            "recommended_ram_gib": 16,
        }
    if "bpc" in datasets:
        required = [
            repo_path(config, config["bpc"]["download"]["reviews_path"]),
            repo_path(config, config["bpc"]["download"]["metadata_path"]),
        ]
        data["bpc"] = {
            "ready": all(path.exists() for path in required),
            "required_files": {
                str(path.relative_to(REPO_ROOT)): path.exists() for path in required
            },
            "recommended_free_disk_gib": 100,
            "recommended_ram_gib": 64,
            "preferred_ram_gib_with_group_control": 96,
        }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Check environment, storage, and dataset readiness.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/paper.yaml"))
    parser.add_argument("--dataset", choices=["movielens", "bpc", "all"], default="all")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON only")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero on an environment mismatch or missing requested data",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    disk = shutil.disk_usage(REPO_ROOT)
    datasets = ["movielens", "bpc"] if args.dataset == "all" else [args.dataset]
    data = dataset_status(config, datasets)

    free_disk_gib = gib(disk.free)
    virtual_disk_report = free_disk_gib > 1024 * 1024
    memory = physical_memory_bytes()
    memory_gib = round(gib(memory), 2) if memory is not None else None
    packages = package_status()
    python_matches = sys.version_info[:2] == (3, 11)
    environment_matches = python_matches and all(
        value["matches"] for value in packages.values()
    )
    data_ready = all(entry["ready"] for entry in data.values())

    warnings: list[str] = []
    for name, entry in data.items():
        if memory_gib is not None and memory_gib < entry["recommended_ram_gib"]:
            warnings.append(
                f"{name}: detected RAM {memory_gib} GiB is below the "
                f"recommended {entry['recommended_ram_gib']} GiB"
            )
        if not virtual_disk_report and free_disk_gib < entry["recommended_free_disk_gib"]:
            warnings.append(
                f"{name}: free disk {free_disk_gib:.2f} GiB is below the "
                f"recommended {entry['recommended_free_disk_gib']} GiB"
            )

    report = {
        "python": {
            "version": platform.python_version(),
            "executable_name": Path(sys.executable).name,
            "matches_paper_environment": python_matches,
            "required": "Python 3.11.x",
        },
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "physical_ram_gib": memory_gib,
            "free_disk_gib": None if virtual_disk_report else round(free_disk_gib, 2),
            "disk_note": (
                "filesystem reports a virtual/unbounded capacity"
                if virtual_disk_report
                else None
            ),
        },
        "packages": packages,
        "datasets": data,
        "warnings": warnings,
        "environment_status": "ready" if environment_matches else "mismatch",
        "data_status": "ready" if data_ready else "download-required",
    }
    report["status"] = (
        "ready"
        if environment_matches and data_ready and not warnings
        else "ready-with-resource-warning"
        if environment_matches and data_ready
        else "download-required"
        if environment_matches
        else "environment-mismatch"
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Preflight status: {report['status']}")
        print(
            f"Python: {report['python']['version']} "
            f"({'paper-compatible' if python_matches else 'expected 3.11.x'})"
        )
        disk_text = (
            report["system"]["disk_note"]
            if report["system"]["free_disk_gib"] is None
            else f"{report['system']['free_disk_gib']} GiB free"
        )
        ram_text = (
            "RAM unknown"
            if memory_gib is None
            else f"{memory_gib} GiB RAM"
        )
        print(f"CPUs: {report['system']['cpu_count']} | {ram_text} | disk: {disk_text}")
        mismatched = [
            f"{name}={entry['actual']} (expected {entry['expected']})"
            for name, entry in packages.items()
            if not entry["matches"]
        ]
        print(
            "Packages: "
            + ("all pinned versions match" if not mismatched else "; ".join(mismatched))
        )
        for name, entry in data.items():
            print(f"{name}: {'data ready' if entry['ready'] else 'download required'}")
        for warning in warnings:
            print(f"WARNING: {warning}")

    if args.strict and (not environment_matches or not data_ready):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
