#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tie_eval.config import load_config
from tie_eval.replay import replay_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay evaluator policies from saved candidate rows and candidate scores."
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/paper.yaml"))
    parser.add_argument("--dataset", choices=["bpc", "movielens", "all"], required=True)
    parser.add_argument("--include-randomized", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    datasets = ["bpc", "movielens"] if args.dataset == "all" else [args.dataset]
    result = {
        dataset: replay_dataset(
            config,
            dataset,
            include_randomized=args.include_randomized,
            include_diagnostics=not args.skip_diagnostics,
        )
        for dataset in datasets
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
