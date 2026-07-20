#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.bpc import run
from tie_eval.config import load_config, repo_path
from tie_eval.download import download_bpc


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the Amazon BPC audit.")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/paper.yaml"))
    parser.add_argument(
        "--stage",
        choices=["stage_raw", "preprocess", "model", "evaluate", "all"],
        default="all",
    )
    parser.add_argument("--download", action="store_true", help="download BPC if needed")
    parser.add_argument("--max-rows", type=int, default=None, help="smoke-test subset; not paper scale")
    parser.add_argument(
        "--skip-low-tie-control",
        action="store_true",
        help="skip the expensive cross-fitted user/group control",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    review_path = repo_path(config, config["bpc"]["download"]["reviews_path"])
    metadata_path = repo_path(config, config["bpc"]["download"]["metadata_path"])
    if args.download or not review_path.exists() or not metadata_path.exists():
        download_bpc(config, force=args.force)
    summary = run(
        config,
        stage=args.stage,
        max_rows=args.max_rows,
        include_low_tie=not args.skip_low_tie_control,
        resume=not args.no_resume,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
