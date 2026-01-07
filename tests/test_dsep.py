import unittest
import networkx as nx

from causal_equiv import is_d_separated


def dag(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


class TestDSeparation(unittest.TestCase):
    # --- Canonical motifs ---

    def test_chain_dependent_unconditioned_and_blocked_by_middle(self):
        # X -> Z -> Y
        G = dag([("X", "Z"), ("Z", "Y")])

        # Unconditioned: active chain path => not d-separated
        self.assertFalse(is_d_separated(G, "X", "Y", set()))

        # Conditioning on non-collider Z blocks chain
        self.assertTrue(is_d_separated(G, "X", "Y", {"Z"}))

    def test_fork_dependent_unconditioned_and_blocked_by_common_cause(self):
        # Z -> X, Z -> Y
        G = dag([("Z", "X"), ("Z", "Y")])

        # Unconditioned: active fork path => not d-separated
        self.assertFalse(is_d_separated(G, "X", "Y", set()))

        # Conditioning on common cause blocks
        self.assertTrue(is_d_separated(G, "X", "Y", {"Z"}))

    def test_collider_independent_unconditioned_and_opened_by_conditioning_on_collider(self):
        # X -> Z <- Y  (collider)
        G = dag([("X", "Z"), ("Y", "Z")])

        # Unconditioned collider blocks
        self.assertTrue(is_d_separated(G, "X", "Y", set()))

        # Conditioning on collider opens
        self.assertFalse(is_d_separated(G, "X", "Y", {"Z"}))

    def test_collider_opened_by_conditioning_on_descendant(self):
        # X -> Z <- Y, Z -> W (W is descendant of collider Z)
        G = dag([("X", "Z"), ("Y", "Z"), ("Z", "W")])

        # Still blocked without conditioning
        self.assertTrue(is_d_separated(G, "X", "Y", set()))

        # Conditioning on descendant of collider opens
        self.assertFalse(is_d_separated(G, "X", "Y", {"W"}))

    # Degenerate / semantics-locking tests (for moralization+delete-Z implementation) 

    def test_condition_on_endpoint_makes_trivially_separated_in_this_implementation(self):
        # With the "moralize then delete Z nodes" implementation,
        # conditioning on an endpoint removes it from the graph, so no path can exist.
        G = dag([("X", "Z"), ("Z", "Y")])
        self.assertTrue(is_d_separated(G, "X", "Y", {"Y"}))
        self.assertTrue(is_d_separated(G, "X", "Y", {"X"}))

    def test_start_equals_end_not_separated(self):
        G = dag([("X", "Y")])
        self.assertFalse(is_d_separated(G, "X", "X", set()))
        self.assertFalse(is_d_separated(G, "X", "X", {"Y"}))

    def test_missing_nodes_treated_as_separated(self):
        G = dag([("A", "B")])
        self.assertTrue(is_d_separated(G, "X", "B", set()))
        self.assertTrue(is_d_separated(G, "A", "Y", {"B"}))

    #  Multiple paths / robustness 

    def test_two_chains_condition_blocks_one_but_other_keeps_dependence(self):
        # X -> Z -> Y and X -> W -> Y (two active chain paths)
        G = dag([("X", "Z"), ("Z", "Y"), ("X", "W"), ("W", "Y")])

        # Unconditioned: dependent
        self.assertFalse(is_d_separated(G, "X", "Y", set()))

        # Conditioning on Z blocks only that path; W path remains
        self.assertFalse(is_d_separated(G, "X", "Y", {"Z"}))

        # Conditioning on both middle nodes blocks both paths
        self.assertTrue(is_d_separated(G, "X", "Y", {"Z", "W"}))

    def test_irrelevant_conditioning_in_disconnected_component_does_not_change_result(self):
        # X -> Z -> Y is one component; U -> V is disconnected
        G = dag([("X", "Z"), ("Z", "Y"), ("U", "V")])

        self.assertFalse(is_d_separated(G, "X", "Y", set()))
        self.assertFalse(is_d_separated(G, "X", "Y", {"U"}))
        self.assertFalse(is_d_separated(G, "X", "Y", {"V"}))
        self.assertFalse(is_d_separated(G, "X", "Y", {"U", "V"}))

    def test_condition_on_non_descendant_of_collider_does_not_open(self):
        # Collider: X -> Z <- Y, plus X -> W (W is NOT a descendant of Z)
        G = dag([("X", "Z"), ("Y", "Z"), ("X", "W")])

        # Unconditioned: blocked by collider
        self.assertTrue(is_d_separated(G, "X", "Y", set()))

        # Conditioning on W should NOT open collider path
        self.assertTrue(is_d_separated(G, "X", "Y", {"W"}))

    def test_ancestor_restriction_irrelevant_parents_of_descendant_do_not_matter(self):
        # X -> Z -> Y, Z -> W, and A -> W (A is not an ancestor of X or Y)
        # Conditioning on W should not change X-Z-Y chain unless it creates a collider situation (it doesn't here).
        G = dag([("X", "Z"), ("Z", "Y"), ("Z", "W"), ("A", "W")])

        # Unconditioned: X and Y dependent via chain
        self.assertFalse(is_d_separated(G, "X", "Y", set()))

        # Conditioning on W (a descendant of Z, but not a collider-descendant scenario for X-Y) shouldn't block the chain
        self.assertFalse(is_d_separated(G, "X", "Y", {"W"}))

        # Conditioning on Z still blocks the chain
        self.assertTrue(is_d_separated(G, "X", "Y", {"Z"}))

    # Symmetry 

    def test_symmetry(self):
        G = dag([("X", "Z"), ("Z", "Y"), ("U", "X")])
        for Zset in [set(), {"Z"}, {"U"}, {"U", "Z"}]:
            self.assertEqual(
                is_d_separated(G, "X", "Y", Zset),
                is_d_separated(G, "Y", "X", Zset),
            )


if __name__ == "__main__":
    unittest.main()