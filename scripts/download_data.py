#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.config import load_config
from tie_eval.download import download_bpc, download_movielens


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public datasets from their official hosts.")
    parser.add_argument("--dataset", choices=["bpc", "movielens", "all"], required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/paper.yaml"))
    parser.add_argument("--force", action="store_true", help="replace existing downloads")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    if args.dataset in {"movielens", "all"}:
        download_movielens(config, force=args.force)
    if args.dataset in {"bpc", "all"}:
        download_bpc(config, force=args.force)


if __name__ == "__main__":
    main()
