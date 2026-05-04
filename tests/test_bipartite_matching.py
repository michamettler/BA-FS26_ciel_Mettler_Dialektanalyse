"""Tests for bipartite_matching."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "domain"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "utils"))

from bipartite_matching import (
    ATTR_WORD,
    EPS,
    HYPOTHESIS_PARTITION,
    REFERENCE_PARTITION,
    SINK_NODE,
    SOURCE_NODE,
    build_full_bipartite_graph,
    build_reduced_graph_by_matching,
    extract_index_from_node_name,
    get_bipartite_edges,
    get_epsilon_edges,
    get_node_name,
    get_nodes_by_partition,
    get_sink_edges,
    get_source_edges,
    get_word_edges,
    is_eps_node,
    solve_matching,
)
from word_similarity_calculator import WordSimilarityCalculator


def ref(i, eps=False):
    return get_node_name(REFERENCE_PARTITION, i, eps=eps)


def hyp(j, eps=False):
    return get_node_name(HYPOTHESIS_PARTITION, j, eps=eps)


class TestGraphTopology(unittest.TestCase):

    # node and edge counts

    def test_node_count_is_two_plus_two_n(self):
        # source + sink + n_r real refs + n_h real hyps + n_h ref-eps + n_r hyp-eps
        calc = WordSimilarityCalculator(sent_len=2)
        G = build_full_bipartite_graph(["a", "b"], ["c", "d"], calc)
        self.assertEqual(len(G.nodes), 2 + 2 * (2 + 2))

    def test_source_and_sink_demands_balance(self):
        # min_cost_flow requires demands to sum to zero; source = -N, sink = +N
        calc = WordSimilarityCalculator(sent_len=2)
        G = build_full_bipartite_graph(["a"], ["b", "c"], calc)
        self.assertEqual(G.nodes[SOURCE_NODE]["demand"], -3)
        self.assertEqual(G.nodes[SINK_NODE]["demand"], 3)

    def test_source_feeds_all_left_nodes(self):
        calc = WordSimilarityCalculator(sent_len=2)
        G = build_full_bipartite_graph(["a", "b"], ["c"], calc)
        self.assertEqual(len(get_source_edges(G)), 2 + 1)

    def test_sink_drains_all_right_nodes(self):
        calc = WordSimilarityCalculator(sent_len=2)
        G = build_full_bipartite_graph(["a", "b"], ["c"], calc)
        self.assertEqual(len(get_sink_edges(G)), 2 + 1)

    def test_bipartite_edges_form_complete_graph(self):
        # both padded sides have N nodes, so the bipartite layer has N^2 edges
        calc = WordSimilarityCalculator(sent_len=2)
        G = build_full_bipartite_graph(["a", "b"], ["c", "d"], calc)
        self.assertEqual(len(get_bipartite_edges(G)), (2 + 2) ** 2)


class TestEdgeAttributes(unittest.TestCase):

    # weights and similarities on bipartite edges

    def test_word_word_edge_weight_is_zero_for_identical_words(self):
        # alpha=1 isolates lexical: identical words give sim=1, so cost = (1-1)*1000 = 0
        calc = WordSimilarityCalculator(sent_len=2, alpha=1.0)
        G = build_full_bipartite_graph(["haus"], ["haus"], calc)
        edge = G[ref(0)][hyp(0)]
        self.assertEqual(edge["weight"], 0)
        self.assertAlmostEqual(edge["similarity"], 1.0)

    def test_word_word_edge_records_partial_similarity(self):
        # Levenshtein.ratio("haus","huus") = 0.75; cost = (1 - 0.75) * 1000 = 250
        calc = WordSimilarityCalculator(sent_len=2, alpha=1.0)
        G = build_full_bipartite_graph(["haus"], ["huus"], calc)
        edge = G[ref(0)][hyp(0)]
        self.assertAlmostEqual(edge["similarity"], 0.75)
        self.assertEqual(edge["weight"], 250)

    def test_epsilon_edges_carry_lambda_cost_and_no_similarity(self):
        calc = WordSimilarityCalculator(sent_len=2, lambda_=0.4)
        G = build_full_bipartite_graph(["a"], ["b"], calc)
        edge = G[ref(0)][hyp(0, eps=True)]
        self.assertEqual(edge["weight"], 400)
        self.assertIsNone(edge["similarity"])

    def test_eps_to_eps_edges_have_zero_weight(self):
        # padding-to-padding edges absorb unused capacity for free
        calc = WordSimilarityCalculator(sent_len=2, lambda_=0.5)
        G = build_full_bipartite_graph(["a"], ["b"], calc)
        edge = G[ref(0, eps=True)][hyp(0, eps=True)]
        self.assertEqual(edge["weight"], 0)
        self.assertIsNone(edge["similarity"])


class TestSolveMatching(unittest.TestCase):

    # min-cost flow on small hand-checked cases

    def test_identical_sentences_pair_diagonally(self):
        calc = WordSimilarityCalculator(sent_len=2, alpha=0.7, lambda_=0.3)
        G = build_full_bipartite_graph(["haus", "garten"], ["haus", "garten"], calc)
        matching = solve_matching(G)
        self.assertEqual(matching[ref(0)], hyp(0))
        self.assertEqual(matching[ref(1)], hyp(1))

    def test_low_lambda_routes_disjoint_words_through_epsilon(self):
        # disjoint words: real-real cost = 1000, two-eps detour cost = 2 * lambda * 1000
        # with lambda=0.1 the detour costs 200 < 1000, so the solver prefers epsilon
        calc = WordSimilarityCalculator(sent_len=2, alpha=1.0, lambda_=0.1)
        G = build_full_bipartite_graph(["abc"], ["xyz"], calc)
        matching = solve_matching(G)
        self.assertTrue(is_eps_node(G.nodes[matching[ref(0)]]))

    def test_high_lambda_pairs_disjoint_words_anyway(self):
        # same disjoint words; lambda=0.6 makes the detour cost 1200 > 1000,
        # so the solver pairs the real words despite zero similarity
        calc = WordSimilarityCalculator(sent_len=2, alpha=1.0, lambda_=0.6)
        G = build_full_bipartite_graph(["abc"], ["xyz"], calc)
        matching = solve_matching(G)
        self.assertEqual(matching[ref(0)], hyp(0))

    def test_extra_ref_words_route_to_hyp_epsilon(self):
        calc = WordSimilarityCalculator(sent_len=3, alpha=1.0, lambda_=0.3)
        G = build_full_bipartite_graph(["haus", "garten", "auto"], ["haus"], calc)
        matching = solve_matching(G)
        self.assertEqual(matching[ref(0)], hyp(0))
        self.assertTrue(is_eps_node(G.nodes[matching[ref(1)]]))
        self.assertTrue(is_eps_node(G.nodes[matching[ref(2)]]))

    def test_matching_covers_every_left_node_exactly_once(self):
        # each ref and ref-eps node must appear as a key once (perfect matching on padded sides)
        calc = WordSimilarityCalculator(sent_len=2, alpha=0.7, lambda_=0.3)
        G = build_full_bipartite_graph(["haus", "garten"], ["huus"], calc)
        matching = solve_matching(G)
        self.assertEqual(len(matching), 2 + 1)
        self.assertEqual(len(set(matching.values())), 2 + 1)


class TestReducedGraph(unittest.TestCase):

    # reduction step keeps the parts useful for plotting and downstream analysis

    def test_excludes_eps_to_eps_pairs(self):
        # diagonal match leaves two eps-eps fillers; reduction must drop them
        calc = WordSimilarityCalculator(sent_len=2, alpha=0.7, lambda_=0.3)
        G = build_full_bipartite_graph(["haus", "garten"], ["haus", "garten"], calc)
        M = build_reduced_graph_by_matching(G, solve_matching(G))
        for u, v in get_bipartite_edges(M):
            self.assertFalse(is_eps_node(M.nodes[u]) and is_eps_node(M.nodes[v]))

    def test_preserves_similarity_on_word_edges(self):
        calc = WordSimilarityCalculator(sent_len=2, alpha=1.0)
        G = build_full_bipartite_graph(["haus"], ["haus"], calc)
        M = build_reduced_graph_by_matching(G, solve_matching(G))
        self.assertAlmostEqual(M[ref(0)][hyp(0)]["similarity"], 1.0)

    def test_includes_source_and_sink_routing_for_real_pairs(self):
        # plotting needs s -> ref -> hyp -> t paths to render the matching
        calc = WordSimilarityCalculator(sent_len=2, alpha=0.7, lambda_=0.3)
        G = build_full_bipartite_graph(["haus"], ["haus"], calc)
        M = build_reduced_graph_by_matching(G, solve_matching(G))
        self.assertIn(SOURCE_NODE, M.nodes)
        self.assertIn(SINK_NODE, M.nodes)
        self.assertTrue(M.has_edge(SOURCE_NODE, ref(0)))
        self.assertTrue(M.has_edge(hyp(0), SINK_NODE))

    def test_eps_routed_matching_yields_no_word_to_word_edges(self):
        # very low lambda routes both sides via eps; no real-real word edge survives
        calc = WordSimilarityCalculator(sent_len=2, alpha=1.0, lambda_=0.05)
        G = build_full_bipartite_graph(["abc"], ["xyz"], calc)
        M = build_reduced_graph_by_matching(G, solve_matching(G))
        self.assertEqual(len(get_word_edges(M, get_bipartite_edges(M))), 0)


class TestHelpers(unittest.TestCase):

    # naming, partition, and edge classification helpers

    def test_node_name_round_trips_through_extract_index(self):
        self.assertEqual(extract_index_from_node_name(get_node_name(REFERENCE_PARTITION, 5)), 5)
        self.assertEqual(extract_index_from_node_name(get_node_name(HYPOTHESIS_PARTITION, 0, eps=True)), 0)
        self.assertEqual(extract_index_from_node_name(get_node_name(REFERENCE_PARTITION, 12, eps=True)), 12)

    def test_eps_flag_changes_node_name(self):
        self.assertNotEqual(
            get_node_name(REFERENCE_PARTITION, 3),
            get_node_name(REFERENCE_PARTITION, 3, eps=True),
        )

    def test_is_eps_node_checks_word_attribute(self):
        self.assertTrue(is_eps_node({ATTR_WORD: EPS}))
        self.assertFalse(is_eps_node({ATTR_WORD: "haus"}))
        self.assertFalse(is_eps_node({}))

    def test_get_nodes_by_partition_includes_real_and_padding(self):
        # ref side has n_r real + n_h padding; hyp side has n_h real + n_r padding
        calc = WordSimilarityCalculator(sent_len=2)
        G = build_full_bipartite_graph(["a", "b"], ["c"], calc)
        self.assertEqual(len(get_nodes_by_partition(G, REFERENCE_PARTITION)), 2 + 1)
        self.assertEqual(len(get_nodes_by_partition(G, HYPOTHESIS_PARTITION)), 1 + 2)

    def test_word_and_epsilon_edge_classifiers_are_complementary(self):
        # every bipartite edge is exactly word OR epsilon - never both, never neither
        calc = WordSimilarityCalculator(sent_len=2)
        G = build_full_bipartite_graph(["a", "b"], ["c"], calc)
        bipartite = get_bipartite_edges(G)
        word_edges = get_word_edges(G, bipartite)
        eps_edges = get_epsilon_edges(G, bipartite)
        self.assertEqual(len(word_edges) + len(eps_edges), len(bipartite))
        self.assertEqual(set(word_edges) & set(eps_edges), set())


if __name__ == "__main__":
    unittest.main()
