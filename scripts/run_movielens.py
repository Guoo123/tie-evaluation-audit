#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.config import load_config, repo_path
from tie_eval.download import download_movielens
from tie_eval.movielens import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the MovieLens Tag Genome audit.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/paper.yaml"))
    parser.add_argument("--download", action="store_true", help="download MovieLens if needed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    data_dir = repo_path(config, config["movielens"]["download"]["extract_dir"])
    if args.download or not (data_dir / "ratings.csv").exists():
        download_movielens(config)
    print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
