#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/preflight.py
python scripts/download_data.py --dataset all --config configs/paper.yaml
python scripts/run_movielens.py --config configs/paper.yaml
python scripts/replay_frozen.py --dataset movielens --config configs/paper.yaml
python scripts/run_bpc.py --config configs/paper.yaml --stage all
python scripts/replay_frozen.py --dataset bpc --config configs/paper.yaml
python scripts/make_paper_outputs.py --config configs/camera_ready.yaml
python scripts/verify_artifact.py --level full --config configs/paper.yaml
