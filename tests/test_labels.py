from __future__ import annotations

import pandas as pd

from tie_eval.labels import BEAUTY_LABELS, apply_target_mask, label_items


def test_beauty_lexicons_match_declared_labels() -> None:
    assert set(BEAUTY_LABELS) == {"fragrance_free", "cruelty_free", "sulfate_free"}
    frame = pd.DataFrame(
        {
            "item_text": [
                "A long product description for an unscented fragrance-free lotion.",
                "A long product description with a Leaping Bunny cruelty-free claim.",
                "A long product description for sulfate-free shampoo with no sulfates.",
                "A long ordinary scented perfume product description.",
            ]
        }
    )
    labeled = label_items(frame, list(BEAUTY_LABELS))
    assert labeled.loc[0, "label__fragrance_free"] == 1
    assert labeled.loc[1, "label__cruelty_free"] == 1
    assert labeled.loc[2, "label__sulfate_free"] == 1
    assert labeled.loc[3, "label__fragrance_free"] == 0


def test_target_terms_are_masked_case_insensitively() -> None:
    text = pd.Series(["CRUELTY-FREE and Leaping Bunny face cream"])
    masked = apply_target_mask(text, BEAUTY_LABELS["cruelty_free"])
    lowered = masked.iloc[0].lower()
    assert "cruelty-free" not in lowered
    assert "leaping bunny" not in lowered
