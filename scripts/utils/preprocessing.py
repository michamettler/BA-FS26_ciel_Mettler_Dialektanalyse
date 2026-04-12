import pandas as pd
import re

def clean_word(text: str) -> str:
    """Clean and normalize text for comparison.

    Lowercases, replaces ß with ss (Swiss German convention), and removes
    all characters except a-z, digits, and German umlauts (äöü).

    Args:
        text: Raw input text to clean.

    Returns:
        Cleaned and normalized string.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("ß", "ss")
    text = re.sub(r"[^a-z0-9äöü\s]", "", text)
    return text.strip()