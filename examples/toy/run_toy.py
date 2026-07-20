#!/usr/bin/env python3
"""Self-contained demonstration of the paper's all-tied failure mode."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tie_eval.metrics import analytic_single_positive_expectation, evaluate_policy
from tie_eval.ranking import TiePolicy


def main() -> None:
    candidates = np.arange(31, dtype=np.int64)[None, :]
    scores = np.zeros((1, 31), dtype=np.float32)
    users = np.array([0], dtype=np.int64)
    positives = np.array([0], dtype=np.int64)
    stable = evaluate_policy(
        candidates,
        scores,
        users,
        positives,
        policy=TiePolicy.STABLE_POSITIVE_FIRST,
        key_seed=20260316,
        k=10,
    )
    hardened = evaluate_policy(
        candidates,
        scores,
        users,
        positives,
        policy=TiePolicy.HARDENED_UINT64,
        key_seed=20260316,
        k=10,
    )
    analytic = analytic_single_positive_expectation(scores, candidates, positives, k=10)
    print(
        json.dumps(
            {
                "stable_positive_first": stable.to_dict(),
                "hardened_uint64": hardened.to_dict(),
                "analytic_uniform_tie_expectation": analytic.to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
