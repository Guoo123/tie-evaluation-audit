from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and retain its source location."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    config["_config_path"] = str(config_path)
    config["_repo_root"] = str(config_path.parent.parent)
    return config


def repo_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a repository-relative path from the loaded config."""
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_repo_root"]) / path
