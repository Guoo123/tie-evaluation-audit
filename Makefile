.PHONY: setup preflight test smoke archive-verify manifest-verify package-check \
        download-movielens run-movielens replay-movielens \
        download-bpc run-bpc replay-bpc run-all verify figure clean-generated

setup:
	python -m pip install -r requirements.txt
	python -m pip install --no-deps -e .

preflight:
	python scripts/preflight.py

test:
	python -m pytest

smoke:
	python scripts/verify_artifact.py --level smoke

archive-verify:
	python scripts/verify_artifact.py --level archive

manifest-verify:
	python scripts/verify_manifest.py

package-check: manifest-verify test archive-verify
	python scripts/check_package_hygiene.py

download-movielens:
	python scripts/download_data.py --dataset movielens

run-movielens:
	python scripts/run_movielens.py --config configs/paper.yaml

replay-movielens:
	python scripts/replay_frozen.py --dataset movielens --config configs/paper.yaml

download-bpc:
	python scripts/download_data.py --dataset bpc

run-bpc:
	python scripts/run_bpc.py --config configs/paper.yaml --stage all

replay-bpc:
	python scripts/replay_frozen.py --dataset bpc --config configs/paper.yaml

run-all:
	python scripts/run_all.py --config configs/paper.yaml

verify:
	python scripts/verify_artifact.py --level full --config configs/paper.yaml

figure:
	python scripts/make_paper_outputs.py --config configs/paper.yaml

clean-generated:
	rm -rf results/regenerated/* data/raw/* data/interim/* data/processed/*
	touch results/regenerated/.gitkeep data/raw/.gitkeep data/interim/.gitkeep data/processed/.gitkeep
