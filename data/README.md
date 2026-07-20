# Local data workspace

No third-party dataset is redistributed in this artifact.

```text
data/raw/        downloaded public archives/files and download manifests
data/interim/    restartable BPC JSONL-to-Parquet staging and DuckDB workspace
data/processed/  processed BPC train/validation/test/items files and count summary
```

Download commands:

```bash
python scripts/download_data.py --dataset movielens
python scripts/download_data.py --dataset bpc
```

All dataset content under the three subdirectories is ignored by Git except the
placeholder files. Upstream data terms continue to apply. See `docs/DATA.md`.
