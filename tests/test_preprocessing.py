"""Tests for preprocessing.clean_word."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from preprocessing import clean_word


class TestCleanWordBasics(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(clean_word("Haus"), "haus")

    def test_eszett_normalised_to_ss(self):
        self.assertEqual(clean_word("Straße"), "strasse")

    def test_umlauts_preserved(self):
        self.assertEqual(clean_word("Junge über"), "junge über")

    def test_punctuation_stripped(self):
        self.assertEqual(clean_word("Hallo, Welt!"), "hallo welt")

    def test_nan_returns_empty_string(self):
        import pandas as pd
        self.assertEqual(clean_word(pd.NA), "")
        self.assertEqual(clean_word(float("nan")), "")

    def test_whitespace_collapsed(self):
        self.assertEqual(clean_word("  zwei   wörter  "), "zwei wörter")

    def test_common_words_uncorrupted(self):
        self.assertEqual(clean_word("Frau"), "frau")
        self.assertEqual(clean_word("laufen"), "laufen")


class TestCleanWordAbbreviations(unittest.TestCase):
    """Spelled-out unit/currency/magnitude words contract to their abbreviation,
    so REF "Zentimeter" and DIT "cm" both produce a "cm" token."""

    def test_zentimeter_and_cm_share_token(self):
        self.assertIn("cm", clean_word("175 Zentimeter gross").split())
        self.assertIn("cm", clean_word("150 cm gross").split())

    def test_kilogramm_and_kg_share_token(self):
        self.assertIn("kg", clean_word("2,6 Kilogramm").split())
        self.assertIn("kg", clean_word("2,6 kg").split())

    def test_kilometer_and_km_share_token(self):
        self.assertIn("km", clean_word("100 Kilometer entfernt").split())
        self.assertIn("km", clean_word("100 km entfernt").split())

    def test_franken_and_fr_share_token(self):
        self.assertIn("fr", clean_word("250 Franken").split())
        self.assertIn("fr", clean_word("250 Fr.").split())

    def test_millionen_and_mio_share_token(self):
        self.assertIn("mio", clean_word("40 Millionen").split())
        self.assertIn("mio", clean_word("40 Mio.").split())

    def test_milliarden_and_mrd_share_token(self):
        self.assertIn("mrd", clean_word("2 Milliarden").split())
        self.assertIn("mrd", clean_word("2 Mrd.").split())

    def test_percent_and_prozent_share_token(self):
        self.assertIn("prozent", clean_word("60%").split())
        self.assertIn("prozent", clean_word("60 Prozent").split())

    def test_quadratmeter_and_m2_share_token(self):
        self.assertIn("m2", clean_word("50 Quadratmeter Wohnfläche").split())
        self.assertIn("m2", clean_word("50 m2 Wohnfläche").split())

    def test_kubikmeter_and_m3_share_token(self):
        self.assertIn("m3", clean_word("10 Kubikmeter Wasser").split())
        self.assertIn("m3", clean_word("10 m3 Wasser").split())

    def test_beziehungsweise_and_bzw_share_token(self):
        self.assertIn("bzw", clean_word("Hund beziehungsweise Katze").split())
        self.assertIn("bzw", clean_word("Hund bzw. Katze").split())


class TestCleanWordDigits(unittest.TestCase):
    """Digits stay in the output (text2num normalises spelled-out forms to
    digits so REF and DIT meet at the digit form)."""

    def test_digits_preserved(self):
        self.assertEqual(clean_word("175 Bücher"), "175 bücher")

    def test_decimal_separator_stripped(self):
        # "2,6" — punctuation strip removes the comma, digits stay.
        self.assertEqual(clean_word("2,6 Liter"), "26 liter")

    def test_pure_number_string_preserved(self):
        self.assertEqual(clean_word("12345"), "12345")


class TestCleanWordSpelledNumerals(unittest.TestCase):
    """text2num normalises spelled-out numerals to digits so REF "hundertsiebzig"
    and DIT "170" both end up as the digit form."""

    def test_compound_numeral_becomes_digits(self):
        self.assertEqual(clean_word("hundertsiebzig Zentimeter"), "170 cm")

    def test_spelled_and_digit_normalise_identically(self):
        self.assertEqual(
            clean_word("hundertsiebzig Bücher"),
            clean_word("170 Bücher"),
        )

    def test_year_form_becomes_digits(self):
        self.assertEqual(clean_word("im Jahr zweitausendfünf"), "im jahr 2005")


if __name__ == "__main__":
    unittest.main()