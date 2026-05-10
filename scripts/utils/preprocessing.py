import re

import pandas as pd
from text_to_num import alpha2digit


def clean_word(word: str) -> str:
    """Clean and normalize a single word for comparison.

    Lowercases, normalizes ß, contracts spelled-out unit/currency/magnitude words
    to their abbreviations (so "Franken" and "Fr." both end up as "fr"), converts
    spelled-out numerals to digits via text2num, and strips punctuation.
    """
    if pd.isna(word):
        return ""

    word = str(word).lower()
    word = word.replace("ß", "ss")
    word = word.replace("%", "prozent")
    word = word.replace("zentimeter", "cm")
    word = word.replace("millimeter", "mm")
    word = word.replace("kilometer", "km")
    word = word.replace("kilogramm", "kg")
    word = word.replace("quadratmeter", "m2")
    word = word.replace("kubikmeter", "m3")
    word = word.replace("milligramm", "mg")
    word = word.replace("milliliter", "ml")
    word = word.replace("millionen", "mio")
    word = word.replace("milliarden", "mrd")
    word = word.replace("beziehungsweise", "bzw")
    word = word.replace("franken", "fr")

    word = alpha2digit(word, "de")
    return re.sub(r"[^a-z0-9äöü]", "", word)