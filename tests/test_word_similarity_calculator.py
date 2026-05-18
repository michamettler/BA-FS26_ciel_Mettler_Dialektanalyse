"""Tests for WordSimilarityCalculator."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))

from word_similarity_calculator import (
    COST_SCALE,
    WordSimilarityCalculator,
    cost_for_word_pair_by_similarity,
    scale_cost_for_networkx,
    similarity_for_word_pair_by_cost,
)


class TestWordSimilarityCalculator(unittest.TestCase):

    # parameter validation

    def test_init_rejects_non_positive_sent_len(self):
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(sent_len=0)
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(sent_len=-1)

    def test_init_rejects_alpha_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(sent_len=5, alpha=-0.1)
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(sent_len=5, alpha=1.5)

    def test_init_rejects_lambda_outside_unit_interval(self):
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(sent_len=5, lambda_=-0.1)
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(sent_len=5, lambda_=1.5)

    def test_init_requires_max_word_len_when_global_normalization_enabled(self):
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(sent_len=5, use_global_lexical_normalization=True)
        with self.assertRaises(ValueError):
            WordSimilarityCalculator(
                sent_len=5, use_global_lexical_normalization=True, max_word_len=0,
            )

    # lexical similarity, local

    def test_lexical_similarity_local_identical_words(self):
        calc = WordSimilarityCalculator(sent_len=5)
        self.assertAlmostEqual(calc.lexical_similarity("haus", "haus"), 1.0)
        self.assertAlmostEqual(calc.lexical_similarity("", ""), 1.0)

    def test_lexical_similarity_local_substitution_exact_value(self):
        # Levenshtein.ratio counts substitutions as cost 2 in (lensum - cost) / lensum
        # "haus" vs. "huus": 1 sub, lensum=8: (8 - 2)/8 = 0.75
        # "hallo" vs. "hello": 1 sub, lensum=10: (10 - 2)/10 = 0.80
        calc = WordSimilarityCalculator(sent_len=5)
        self.assertAlmostEqual(calc.lexical_similarity("haus", "huus"), 0.75)
        self.assertAlmostEqual(calc.lexical_similarity("hallo", "hello"), 0.80)

    def test_lexical_similarity_prefers_morphological_match_over_unrelated_word(self):
        calc = WordSimilarityCalculator(sent_len=5)
        self.assertGreater(calc.lexical_similarity("zog", "gezogen"),
                            calc.lexical_similarity("zog", "hat"))

        print(calc.lexical_similarity("zog", "gezogen"))
        print(calc.lexical_similarity("zog", "hat"))

    def test_lexical_similarity_local_insertion_exact_value(self):
        # insertions count as cost 1: "haus" vs. "haust": distance=1, lensum=9: (9 - 1)/9
        calc = WordSimilarityCalculator(sent_len=5)
        self.assertAlmostEqual(calc.lexical_similarity("haus", "haust"), 8 / 9)

    def test_lexical_similarity_local_completely_different_words(self):
        # "abc" vs. "xyz": 3 subs, lensum=6: (6 - 6)/6 = 0.0
        calc = WordSimilarityCalculator(sent_len=5)
        self.assertAlmostEqual(calc.lexical_similarity("abc", "xyz"), 0.0)

    # lexical similarity, global

    def test_lexical_similarity_global_identical_words(self):
        calc = WordSimilarityCalculator(
            sent_len=5, use_global_lexical_normalization=True, max_word_len=10,
        )
        self.assertAlmostEqual(calc.lexical_similarity("haus", "haus"), 1.0)

    def test_lexical_similarity_global_exact_values(self):
        calc = WordSimilarityCalculator(
            sent_len=5, use_global_lexical_normalization=True, max_word_len=10,
        )
        self.assertAlmostEqual(calc.lexical_similarity("haus", "huus"), 0.9)  # d=1, 1 - 1/10
        self.assertAlmostEqual(calc.lexical_similarity("abc", "xyz"), 0.7)  # d=3, 1 - 3/10
        self.assertAlmostEqual(calc.lexical_similarity("hallo", "world"), 0.6)  # d=4, 1 - 4/10

    def test_lexical_similarity_global_normalization_scales_with_max_word_len(self):
        # same word pair, different max_word_len: similarity scales accordingly
        calc_short = WordSimilarityCalculator(
            sent_len=5, use_global_lexical_normalization=True, max_word_len=5,
        )
        calc_long = WordSimilarityCalculator(
            sent_len=5, use_global_lexical_normalization=True, max_word_len=20,
        )
        # "haus" vs "huus" distance=1
        self.assertAlmostEqual(calc_short.lexical_similarity("haus", "huus"), 0.8)  # 1 - 1/5
        self.assertAlmostEqual(calc_long.lexical_similarity("haus", "huus"), 0.95)  # 1 - 1/20

    # positional similarity (linear)

    def test_positional_similarity_linear_endpoints(self):
        # sent_len=11: max_distance=10
        calc = WordSimilarityCalculator(sent_len=11)
        self.assertEqual(calc.positional_similarity(0, 0), 1.0)
        self.assertEqual(calc.positional_similarity(0, 10), 0.0)

    def test_positional_similarity_linear_exact_values(self):
        # sent_len=11: max_distance=10; sim = 1 - gap/10
        calc = WordSimilarityCalculator(sent_len=11)
        self.assertAlmostEqual(calc.positional_similarity(0, 1), 0.9)
        self.assertAlmostEqual(calc.positional_similarity(0, 2), 0.8)
        self.assertAlmostEqual(calc.positional_similarity(0, 5), 0.5)
        self.assertAlmostEqual(calc.positional_similarity(0, 8), 0.2)

    def test_positional_similarity_normalization_scales_with_sent_len(self):
        # same absolute gap, different sentence lengths: different similarities
        short = WordSimilarityCalculator(sent_len=5)  # max_distance=4
        long_ = WordSimilarityCalculator(sent_len=11)  # max_distance=10
        # gap=2: short: 1 - 2/4 = 0.5; long: 1 - 2/10 = 0.8
        self.assertAlmostEqual(short.positional_similarity(0, 2), 0.5)
        self.assertAlmostEqual(long_.positional_similarity(0, 2), 0.8)

    def test_positional_similarity_symmetric_in_indices(self):
        # |i - j| is symmetric, so positional_similarity(i, j) == positional_similarity(j, i)
        calc = WordSimilarityCalculator(sent_len=11)
        self.assertEqual(calc.positional_similarity(2, 7), calc.positional_similarity(7, 2))

    def test_positional_similarity_trivial_sentence(self):
        calc = WordSimilarityCalculator(sent_len=1)
        self.assertEqual(calc.positional_similarity(0, 0), 1.0)

    # positional similarity (squared)

    def test_positional_similarity_squared_endpoints_unchanged(self):
        calc = WordSimilarityCalculator(sent_len=11, use_squared_positional=True)
        self.assertEqual(calc.positional_similarity(0, 0), 1.0)
        self.assertEqual(calc.positional_similarity(0, 10), 0.0)

    def test_positional_similarity_squared_exact_values(self):
        # sent_len=11: max_distance=10; sim = (1 - gap/10)²
        calc = WordSimilarityCalculator(sent_len=11, use_squared_positional=True)
        self.assertAlmostEqual(calc.positional_similarity(0, 1), 0.81)  # 0.9^2
        self.assertAlmostEqual(calc.positional_similarity(0, 2), 0.64)  # 0.8^2
        self.assertAlmostEqual(calc.positional_similarity(0, 5), 0.25)  # 0.5^2
        self.assertAlmostEqual(calc.positional_similarity(0, 8), 0.04)  # 0.2^2

    def test_positional_similarity_squared_strictly_below_linear_at_interior_points(self):
        linear = WordSimilarityCalculator(sent_len=11)
        squared = WordSimilarityCalculator(sent_len=11, use_squared_positional=True)
        for gap in range(1, 10):
            self.assertLess(
                squared.positional_similarity(0, gap),
                linear.positional_similarity(0, gap),
                f"squared should be strictly less than linear at gap={gap}",
            )

    # combined similarity

    def test_combined_similarity_alpha_one_returns_lexical_only(self):
        calc = WordSimilarityCalculator(sent_len=11, alpha=1.0)
        # identical words at far positions: positional would be 0, lexical 1.0: α=1 picks lexical
        self.assertAlmostEqual(
            calc.combined_weighted_lexical_positional_similarity("haus", 0, "haus", 10),
            1.0,
        )

    def test_combined_similarity_alpha_zero_returns_positional_only(self):
        calc = WordSimilarityCalculator(sent_len=11, alpha=0.0)
        # identical words at far positions: positional 0, lexical 1: α=0 picks positional
        self.assertAlmostEqual(
            calc.combined_weighted_lexical_positional_similarity("haus", 0, "haus", 10),
            0.0,
        )

    def test_combined_similarity_weighted_average(self):
        calc = WordSimilarityCalculator(sent_len=11, alpha=0.5)
        # identical words (lex 1.0) at midpoint (pos 0.5): 0.5*1 + 0.5*0.5 = 0.75
        self.assertAlmostEqual(
            calc.combined_weighted_lexical_positional_similarity("haus", 0, "haus", 5),
            0.75,
        )

    # word-pair cost (analytical, float)

    def test_cost_for_word_pair_is_zero_when_similarity_is_one(self):
        self.assertEqual(cost_for_word_pair_by_similarity(1.0), 0.0)

    def test_cost_for_word_pair_is_one_when_similarity_is_zero(self):
        self.assertEqual(cost_for_word_pair_by_similarity(0.0), 1.0)

    def test_cost_for_word_pair_is_one_minus_similarity(self):
        self.assertAlmostEqual(cost_for_word_pair_by_similarity(0.75), 0.25)
        self.assertAlmostEqual(cost_for_word_pair_by_similarity(0.5), 0.5)
        self.assertAlmostEqual(cost_for_word_pair_by_similarity(0.1), 0.9)

    # similarity
    
    def test_similarity_for_word_pair_is_one_minus_cost(self):
        self.assertAlmostEqual(similarity_for_word_pair_by_cost(0.25), 0.75)
        self.assertAlmostEqual(similarity_for_word_pair_by_cost(0.5), 0.5)
        self.assertAlmostEqual(similarity_for_word_pair_by_cost(0.9), 0.1)

    def test_cost_and_similarity_converters_are_mutual_inverses(self):
        for x in (0.0, 0.1, 0.45, 0.55, 1.0):
            self.assertAlmostEqual(similarity_for_word_pair_by_cost(cost_for_word_pair_by_similarity(x)), x)
            self.assertAlmostEqual(cost_for_word_pair_by_similarity(similarity_for_word_pair_by_cost(x)), x)

    # networkx int scaling

    def test_scale_cost_zero_stays_zero(self):
        self.assertEqual(scale_cost_for_networkx(0.0), 0)

    def test_scale_cost_one_maps_to_full_scale(self):
        self.assertEqual(scale_cost_for_networkx(1.0), COST_SCALE)

    def test_scale_cost_multiplies_by_cost_scale(self):
        self.assertEqual(scale_cost_for_networkx(0.25), 250)
        self.assertEqual(scale_cost_for_networkx(0.5), 500)
        self.assertEqual(scale_cost_for_networkx(0.9), 900)

    def test_scale_cost_rounds_to_nearest_int(self):
        # 0.1234 * 1000 = 123.4, rounds to 123
        self.assertEqual(scale_cost_for_networkx(0.1234), 123)
        # 0.1236 * 1000 = 123.6, rounds up to 124
        self.assertEqual(scale_cost_for_networkx(0.1236), 124)

    def test_scale_cost_returns_int_type(self):
        # network simplex demands ints; a float with .0 wouldn't satisfy strict typing
        self.assertIsInstance(scale_cost_for_networkx(0.5), int)

    # epsilon cost (returns un-scaled λ in [0, 1]; scaling for networkx is the caller's job)

    def test_cost_for_epsilon_returns_lambda(self):
        self.assertAlmostEqual(WordSimilarityCalculator(sent_len=5, lambda_=0.0).cost_for_epsilon_by_penalty(), 0.0)
        self.assertAlmostEqual(WordSimilarityCalculator(sent_len=5, lambda_=1.0).cost_for_epsilon_by_penalty(), 1.0)
        self.assertAlmostEqual(WordSimilarityCalculator(sent_len=5, lambda_=0.3).cost_for_epsilon_by_penalty(), 0.3)
        self.assertAlmostEqual(WordSimilarityCalculator(sent_len=5, lambda_=0.45).cost_for_epsilon_by_penalty(), 0.45)

    def test_word_pair_and_epsilon_costs_share_cost_domain(self):
        # both return un-scaled costs in [0, 1]; cost(sim=0.6) == cost_for_epsilon when λ=0.4
        calc = WordSimilarityCalculator(sent_len=5, lambda_=0.4)
        self.assertAlmostEqual(
            calc.cost_for_epsilon_by_penalty(),
            cost_for_word_pair_by_similarity(0.6),
        )


if __name__ == "__main__":
    unittest.main()
