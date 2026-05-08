import re

import pandas as pd
from text_to_num import alpha2digit


def clean_word(text: str) -> str:
    """Clean and normalize text for comparison.

    Lowercases, normalizes ß and %, contracts spelled-out unit/currency/magnitude
    words to their abbreviations (so "Franken" and "Fr." both end up as "fr"),
    converts spelled-out numerals to digits via text2num, and strips punctuation.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("ß", "ss")
    text = text.replace("%", " prozent ")
    text = text.replace("zentimeter", "cm")
    text = text.replace("millimeter", "mm")
    text = text.replace("kilometer", "km")
    text = text.replace("kilogramm", "kg")
    text = text.replace("milligramm", "mg")
    text = text.replace("milliliter", "ml")
    text = text.replace("millionen", "mio")
    text = text.replace("milliarden", "mrd")
    text = text.replace("franken", "fr")

    text = alpha2digit(text, "de")
    text = re.sub(r"[^a-z0-9äöü\s]", "", text)
    return " ".join(text.split())
