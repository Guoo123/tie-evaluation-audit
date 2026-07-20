from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    name: str
    positive_patterns: tuple[str, ...]
    negative_patterns: tuple[str, ...]
    mask_terms: tuple[str, ...]


BEAUTY_LABELS: dict[str, LabelSpec] = {
    "fragrance_free": LabelSpec(
        "fragrance_free",
        (r"\bfragrance[- ]?free\b", r"\bunscented\b", r"\bno fragrance\b", r"\bwithout fragrance\b"),
        (r"\bfragrance\b", r"\bscented\b", r"\bperfume\b"),
        ("fragrance free", "fragrance-free", "unscented", "no fragrance"),
    ),
    "cruelty_free": LabelSpec(
        "cruelty_free",
        (r"\bcruelty[- ]?free\b", r"\bnot tested on animals\b", r"\bleaping bunny\b"),
        (),
        ("cruelty free", "cruelty-free", "not tested on animals", "leaping bunny"),
    ),
    "sulfate_free": LabelSpec(
        "sulfate_free",
        (r"\bsulfate[- ]?free\b", r"\bno sulfates\b", r"\bwithout sulfates\b", r"\bsls[- ]?free\b"),
        (r"\bsulfate\b",),
        ("sulfate free", "sulfate-free", "no sulfates", "sls free", "sls-free"),
    ),
}


def _compile_union(patterns: Iterable[str]) -> re.Pattern[str] | None:
    values = [pattern for pattern in patterns if pattern]
    if not values:
        return None
    return re.compile("|".join(f"(?:{value})" for value in values), re.I)


def label_items(items: pd.DataFrame, label_names: list[str], unknown_as_negative: bool = True) -> pd.DataFrame:
    output = items.copy()
    text = output["item_text"].fillna("").astype(str)
    enough_text = text.str.len() >= 20
    for name in label_names:
        spec = BEAUTY_LABELS[name]
        positive_pattern = _compile_union(spec.positive_patterns)
        negative_pattern = _compile_union(spec.negative_patterns)
        positive = text.str.contains(positive_pattern) if positive_pattern else pd.Series(False, index=output.index)
        negative = text.str.contains(negative_pattern) if negative_pattern else pd.Series(False, index=output.index)
        values = np.where(positive, 1, 0)
        if negative_pattern is not None:
            values = np.where(negative & ~positive, 0, values)
        if not unknown_as_negative:
            values = np.where(enough_text, values, np.nan)
        output[f"label__{name}"] = values
    return output


def apply_target_mask(text: pd.Series, spec: LabelSpec) -> pd.Series:
    masked = text.fillna("").astype(str)
    for term in spec.mask_terms:
        masked = masked.str.replace(term, " ", regex=False, case=False)
    return masked.str.replace(r"\s+", " ", regex=True).str.strip()
