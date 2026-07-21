from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import orjson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import repo_path
from .labels import label_items
from .provenance import write_json
from .text import flatten_textish, normalize_text, parse_details, parse_price

LOGGER = logging.getLogger(__name__)

REVIEW_SCHEMA = pa.schema(
    [
        ("source_row", pa.int64()),
        ("user_id", pa.string()),
        ("item_id", pa.string()),
        ("rating", pa.float64()),
        ("timestamp", pa.int64()),
    ]
)

META_SCHEMA = pa.schema(
    [
        ("source_row", pa.int64()),
        ("item_id", pa.string()),
        ("title", pa.string()),
        ("subtitle", pa.string()),
        ("author", pa.string()),
        ("store", pa.string()),
        ("main_category", pa.string()),
        ("categories_json", pa.string()),
        ("price_num", pa.float64()),
        ("item_text", pa.string()),
    ]
)


def _as_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _as_timestamp(value: Any) -> int | None:
    number = _as_number(value)
    return int(number) if number is not None else None


def _json_string(value: Any) -> str:
    try:
        return orjson.dumps(value if value is not None else []).decode("utf-8")
    except TypeError:
        return json.dumps(value if value is not None else [], ensure_ascii=False, default=str)


def _review_record(obj: dict[str, Any], source_row: int) -> dict[str, Any] | None:
    item = obj.get("parent_asin")
    user = obj.get("user_id")
    rating = _as_number(obj.get("rating"))
    timestamp = _as_timestamp(obj.get("timestamp"))
    if item is None or user is None or rating is None or timestamp is None:
        return None
    return {
        "source_row": source_row,
        "user_id": str(user),
        "item_id": str(item),
        "rating": rating,
        "timestamp": timestamp,
    }


def _meta_record(obj: dict[str, Any], source_row: int) -> dict[str, Any]:
    def text_value(name: str) -> str:
        value = obj.get(name)
        return "" if value is None else str(value)

    details = parse_details(obj.get("details"))
    categories = obj.get("categories") if obj.get("categories") is not None else []
    parts = [
        text_value("title"),
        text_value("subtitle"),
        text_value("author"),
        text_value("store"),
        text_value("main_category"),
        flatten_textish(categories),
        flatten_textish(obj.get("features")),
        flatten_textish(obj.get("description")),
        flatten_textish(details),
    ]
    item_text = normalize_text(" ".join(part for part in parts if part))
    return {
        "source_row": source_row,
        "item_id": str(obj.get("parent_asin")),
        "title": text_value("title"),
        "subtitle": text_value("subtitle"),
        "author": text_value("author"),
        "store": text_value("store"),
        "main_category": text_value("main_category"),
        "categories_json": _json_string(categories),
        "price_num": parse_price(obj.get("price")),
        "item_text": item_text,
    }


def _stage_jsonl(
    source: Path,
    destination: Path,
    *,
    kind: str,
    chunk_rows: int = 100_000,
    force: bool = False,
) -> dict[str, int]:
    if destination.exists() and not force:
        metadata = pq.ParquetFile(destination).metadata
        return {"raw_rows": -1, "staged_rows": metadata.num_rows}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    schema = REVIEW_SCHEMA if kind == "reviews" else META_SCHEMA
    writer = pq.ParquetWriter(temporary, schema=schema, compression="zstd")
    raw_rows = 0
    staged_rows = 0
    batch: list[dict[str, Any]] = []
    try:
        with gzip.open(source, "rb") as handle:
            for source_row, line in enumerate(tqdm(handle, desc=f"stage {kind}", unit=" rows")):
                raw_rows += 1
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                record = (
                    _review_record(obj, source_row)
                    if kind == "reviews"
                    else _meta_record(obj, source_row)
                )
                if record is None:
                    continue
                batch.append(record)
                if len(batch) >= chunk_rows:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                    staged_rows += len(batch)
                    batch.clear()
            if batch:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                staged_rows += len(batch)
    finally:
        writer.close()
    temporary.replace(destination)
    return {"raw_rows": raw_rows, "staged_rows": staged_rows}


def stage_raw_files(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    download = config["bpc"]["download"]
    review_source = repo_path(config, download["reviews_path"])
    meta_source = repo_path(config, download["metadata_path"])
    interim = repo_path(config, "data/interim/bpc")
    review_destination = interim / "reviews_staged.parquet"
    meta_destination = interim / "metadata_staged.parquet"
    if not review_source.exists() or not meta_source.exists():
        raise FileNotFoundError("BPC raw files are missing. Run scripts/download_data.py --dataset bpc")
    manifest = {
        "reviews": _stage_jsonl(review_source, review_destination, kind="reviews", force=force),
        "metadata": _stage_jsonl(meta_source, meta_destination, kind="metadata", force=force),
        "outputs": {
            "reviews": str(review_destination),
            "metadata": str(meta_destination),
        },
    }
    write_json(manifest, interim / "stage_manifest.json")
    return manifest


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def preprocess(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    """Canonicalize, 3-core filter, index, and split BPC using DuckDB.

    The SQL formulation is a bounded-memory translation of the archived pandas
    preprocessing.  It preserves source-row information for duplicate resolution
    and explicitly orders every exported split.
    """
    interim = repo_path(config, "data/interim/bpc")
    processed = repo_path(config, "data/processed/bpc")
    processed.mkdir(parents=True, exist_ok=True)
    expected_outputs = [
        processed / "train.parquet",
        processed / "validation.parquet",
        processed / "test.parquet",
        processed / "items.parquet",
        processed / "dataset_summary.json",
    ]
    if all(path.exists() for path in expected_outputs) and not force:
        with (processed / "dataset_summary.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    review_path = interim / "reviews_staged.parquet"
    meta_path = interim / "metadata_staged.parquet"
    if not review_path.exists() or not meta_path.exists():
        stage_raw_files(config, force=force)

    database = interim / "preprocess.duckdb"
    if force:
        database.unlink(missing_ok=True)
    con = duckdb.connect(str(database))
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=true")
    prep = config["bpc"]["preprocessing"]
    try:
        con.execute("DROP TABLE IF EXISTS canonical_reviews")
        con.execute(
            f"""
            CREATE TABLE canonical_reviews AS
            SELECT user_id, item_id, rating, timestamp, source_row,
                   row_number() OVER (ORDER BY user_id, timestamp, item_id, source_row) AS row_order
            FROM (
              SELECT *, row_number() OVER (
                PARTITION BY user_id, item_id, timestamp ORDER BY source_row DESC
              ) AS duplicate_rank
              FROM read_parquet('{_sql_path(review_path)}')
            )
            WHERE duplicate_rank = 1
            ORDER BY user_id, timestamp, item_id, source_row
            """
        )
        con.execute("DROP TABLE IF EXISTS canonical_meta")
        con.execute(
            f"""
            CREATE TABLE canonical_meta AS
            SELECT * EXCLUDE (duplicate_rank)
            FROM (
              SELECT *, row_number() OVER (
                PARTITION BY item_id ORDER BY source_row DESC
              ) AS duplicate_rank
              FROM read_parquet('{_sql_path(meta_path)}')
            )
            WHERE duplicate_rank = 1
            """
        )
        con.execute("DROP TABLE IF EXISTS core")
        con.execute(
            """
            CREATE TABLE core AS
            SELECT r.*
            FROM canonical_reviews r
            INNER JOIN (SELECT DISTINCT item_id FROM canonical_meta) m USING (item_id)
            ORDER BY r.row_order
            """
        )

        min_users = int(prep["min_user_interactions"])
        min_items = int(prep["min_item_interactions"])
        history: list[dict[str, int]] = []
        for iteration in range(int(prep["max_kcore_iterations"])):
            before = int(con.execute("SELECT count(*) FROM core").fetchone()[0])
            con.execute("DROP TABLE IF EXISTS core_user")
            con.execute(
                f"""
                CREATE TABLE core_user AS
                SELECT c.* FROM core c
                INNER JOIN (
                  SELECT user_id FROM core GROUP BY user_id HAVING count(*) >= {min_users}
                ) u USING (user_id)
                ORDER BY row_order
                """
            )
            con.execute("DROP TABLE IF EXISTS core_next")
            con.execute(
                f"""
                CREATE TABLE core_next AS
                SELECT c.* FROM core_user c
                INNER JOIN (
                  SELECT item_id FROM core_user GROUP BY item_id HAVING count(*) >= {min_items}
                ) i USING (item_id)
                ORDER BY row_order
                """
            )
            after = int(con.execute("SELECT count(*) FROM core_next").fetchone()[0])
            history.append({"iteration": iteration + 1, "before": before, "after": after})
            con.execute("DROP TABLE core")
            con.execute("ALTER TABLE core_next RENAME TO core")
            con.execute("DROP TABLE core_user")
            if after == before:
                break

        con.execute("DROP TABLE IF EXISTS user_map")
        con.execute(
            """
            CREATE TABLE user_map AS
            SELECT user_id, row_number() OVER (ORDER BY user_id) - 1 AS user_idx
            FROM (SELECT DISTINCT user_id FROM core)
            """
        )
        con.execute("DROP TABLE IF EXISTS item_map")
        con.execute(
            """
            CREATE TABLE item_map AS
            SELECT item_id, row_number() OVER (ORDER BY item_id) - 1 AS item_idx
            FROM (SELECT DISTINCT item_id FROM core)
            """
        )
        con.execute("DROP TABLE IF EXISTS mapped")
        con.execute(
            """
            CREATE TABLE mapped AS
            SELECT u.user_idx::BIGINT AS user_idx,
                   i.item_idx::BIGINT AS item_idx,
                   c.user_id, c.item_id, c.rating, c.timestamp, c.row_order,
                   greatest(0.0, least(1.0, (c.rating - 1.0) / 4.0))::FLOAT AS y
            FROM core c
            JOIN user_map u USING (user_id)
            JOIN item_map i USING (item_id)
            ORDER BY c.row_order
            """
        )
        con.execute("DROP TABLE IF EXISTS ranked")
        con.execute(
            """
            CREATE TABLE ranked AS
            SELECT *,
              row_number() OVER (PARTITION BY user_idx ORDER BY timestamp, item_id) AS within_user_rank,
              count(*) OVER (PARTITION BY user_idx) AS within_user_count
            FROM mapped
            """
        )

        def copy_split(name: str, predicate: str) -> None:
            destination = processed / f"{name}.parquet"
            destination.unlink(missing_ok=True)
            con.execute(
                f"""
                COPY (
                  SELECT user_idx, item_idx, rating, timestamp, y
                  FROM ranked WHERE {predicate}
                  ORDER BY user_idx, within_user_rank
                ) TO '{_sql_path(destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)
                """
            )

        copy_split("train", "within_user_rank <= within_user_count - 2")
        copy_split("validation", "within_user_rank = within_user_count - 1")
        copy_split("test", "within_user_rank = within_user_count")

        items_destination = processed / "items_base.parquet"
        items_destination.unlink(missing_ok=True)
        con.execute(
            f"""
            COPY (
              WITH stats AS (
                SELECT item_id,
                       count(*) AS interaction_count,
                       count(DISTINCT user_id) AS user_count,
                       avg(rating) AS mean_review_rating,
                       max(timestamp) AS latest_timestamp
                FROM core GROUP BY item_id
              )
              SELECT im.item_idx::BIGINT AS item_idx,
                     m.item_id, m.title, m.subtitle, m.author, m.store,
                     m.main_category, m.categories_json, m.price_num, m.item_text,
                     s.interaction_count::BIGINT AS interaction_count,
                     s.user_count::BIGINT AS user_count,
                     s.mean_review_rating,
                     s.latest_timestamp::BIGINT AS latest_timestamp
              FROM item_map im
              JOIN canonical_meta m USING (item_id)
              JOIN stats s USING (item_id)
              ORDER BY im.item_idx
            ) TO '{_sql_path(items_destination)}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        items = pd.read_parquet(items_destination)
        items["categories"] = items["categories_json"].map(json.loads)
        items = label_items(
            items,
            list(config["bpc"]["labels"]["names"]),
            bool(config["bpc"]["labels"]["unknown_as_negative"]),
        )
        items.to_parquet(processed / "items.parquet", index=False)

        counts = {
            "n_users": int(con.execute("SELECT count(*) FROM user_map").fetchone()[0]),
            "n_items": int(con.execute("SELECT count(*) FROM item_map").fetchone()[0]),
            "n_train": int(con.execute("SELECT count(*) FROM ranked WHERE within_user_rank <= within_user_count - 2").fetchone()[0]),
            "n_validation": int(con.execute("SELECT count(*) FROM ranked WHERE within_user_rank = within_user_count - 1").fetchone()[0]),
            "n_test": int(con.execute("SELECT count(*) FROM ranked WHERE within_user_rank = within_user_count").fetchone()[0]),
            "kcore_history": history,
            "label_names": list(config["bpc"]["labels"]["names"]),
        }
        write_json(counts, processed / "dataset_summary.json")
        return counts
    finally:
        con.close()
