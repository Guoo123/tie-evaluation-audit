from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

from .provenance import digest, write_json


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe zip member: {member.filename}")
        handle.extractall(destination)


def download_file(
    url: str,
    destination: str | Path,
    *,
    expected_hash: str | None = None,
    hash_algorithm: str = "sha256",
    force: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    """Stream a public file to disk, verify it, and return a provenance record."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        actual = digest(destination, hash_algorithm)
        if expected_hash is None or actual.lower() == expected_hash.lower():
            return {
                "url": url,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                hash_algorithm: actual,
                "status": "reused",
            }
        raise ValueError(
            f"Existing file checksum mismatch for {destination}; use --force to replace it"
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        hasher = hashlib.new(hash_algorithm)
        with temporary.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                hasher.update(chunk)
                progress.update(len(chunk))
    actual_hash = hasher.hexdigest()
    if expected_hash is not None and actual_hash.lower() != expected_hash.lower():
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"Checksum mismatch for {url}: expected {expected_hash}, got {actual_hash}"
        )
    os.replace(temporary, destination)
    return {
        "url": url,
        "path": str(destination),
        "bytes": destination.stat().st_size,
        hash_algorithm: actual_hash,
        "status": "downloaded",
    }


def download_movielens(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"])
    settings = config["movielens"]["download"]
    archive = repo_root / settings["zip_path"]
    md5_path = repo_root / settings["md5_path"]
    extract_dir = repo_root / settings["extract_dir"]

    checksum_record = download_file(
        settings["md5_url"],
        md5_path,
        force=force,
    )
    checksum_text = md5_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\b([0-9a-fA-F]{32})\b", checksum_text)
    if match is None:
        raise ValueError(f"Could not parse an MD5 digest from {md5_path}")
    published_md5 = match.group(1).lower()
    pinned_md5 = str(settings.get("expected_md5", published_md5)).lower()
    if published_md5 != pinned_md5:
        raise ValueError(
            "The current GroupLens checksum differs from the pinned paper checksum: "
            f"published={published_md5}, pinned={pinned_md5}"
        )
    record = download_file(
        settings["zip_url"],
        archive,
        expected_hash=published_md5,
        hash_algorithm="md5",
        force=force,
    )
    required = ["ratings.csv", "genome-tags.csv", "genome-scores.csv"]
    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    if not all((extract_dir / name).exists() for name in required):
        parent = extract_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary:
            temporary_path = Path(temporary)
            _safe_extract_zip(archive, temporary_path)
            extracted = temporary_path / "ml-25m"
            if not extracted.exists():
                raise FileNotFoundError("ml-25m directory was not present in the official archive")
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            shutil.move(str(extracted), str(extract_dir))
    manifest = {
        "dataset": "MovieLens 25M",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "published_checksum_file": checksum_record,
        "published_md5": published_md5,
        "archive": record,
        "files": [
            {
                "path": str(extract_dir / name),
                "bytes": (extract_dir / name).stat().st_size,
                "sha256": digest(extract_dir / name),
            }
            for name in required
        ],
    }
    write_json(manifest, repo_root / "data/raw/movielens/download_manifest.json")
    return manifest


def download_bpc(config: dict[str, Any], *, force: bool = False) -> dict[str, Any]:
    repo_root = Path(config["_repo_root"])
    settings = config["bpc"]["download"]
    review_path = repo_root / settings["reviews_path"]
    metadata_path = repo_root / settings["metadata_path"]
    records = [
        download_file(settings["reviews_url"], review_path, force=force),
        download_file(settings["metadata_url"], metadata_path, force=force),
    ]
    manifest = {
        "dataset": "Amazon Reviews 2023 / Beauty and Personal Care",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "note": (
            "Amazon does not publish a checksum next to these category files. "
            "This manifest records the SHA-256 observed by the regenerating run."
        ),
    }
    write_json(manifest, repo_root / "data/raw/bpc/download_manifest.json")
    return manifest
