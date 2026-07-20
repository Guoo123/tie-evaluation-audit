#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.bpc import run as run_bpc
from tie_eval.config import load_config, repo_path
from tie_eval.download import download_bpc, download_movielens
from tie_eval.movielens import run as run_movielens


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate both paper audits.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/paper.yaml"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-bpc", action="store_true")
    parser.add_argument("--skip-movielens", action="store_true")
    parser.add_argument("--bpc-max-rows", type=int, default=None)
    parser.add_argument("--skip-low-tie-control", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    output: dict[str, object] = {}

    if not args.skip_movielens:
        movie_dir = repo_path(config, config["movielens"]["download"]["extract_dir"])
        if args.download or not (movie_dir / "ratings.csv").exists():
            download_movielens(config)
        output["movielens"] = run_movielens(config)

    if not args.skip_bpc:
        review_path = repo_path(config, config["bpc"]["download"]["reviews_path"])
        meta_path = repo_path(config, config["bpc"]["download"]["metadata_path"])
        if args.download or not review_path.exists() or not meta_path.exists():
            download_bpc(config)
        output["bpc"] = run_bpc(
            config,
            stage="all",
            max_rows=args.bpc_max_rows,
            include_low_tie=not args.skip_low_tie_control,
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
