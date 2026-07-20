from __future__ import annotations

import json
import math
import re
from typing import Any

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower().replace("\n", " ").replace("\t", " ")
    return _WHITESPACE.sub(" ", text)


def flatten_textish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_textish(v) for v in value if v is not None)
    if isinstance(value, dict):
        return " ".join(f"{k} {flatten_textish(v)}" for k, v in value.items())
    return str(value)


def parse_details(details: Any) -> dict[str, Any]:
    if details is None:
        return {}
    if isinstance(details, dict):
        return details
    if isinstance(details, str):
        stripped = details.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {"raw_details": stripped}
    return {"raw_details": str(details)}


def parse_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip().replace("$", "").replace(",", "")
    if text.lower() in {"", "none", "nan", "null"}:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(match.group(1)) if match else None


def price_bucket(price: float | None) -> str:
    if price is None or (isinstance(price, float) and math.isnan(price)):
        return "price::unknown"
    for bound in (10, 20, 35, 50, 75, 100):
        if price <= bound:
            return f"price::<={bound}"
    return "price::>100"
