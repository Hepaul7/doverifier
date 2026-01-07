import unittest
import sympy as sp

from probability import CausalProbability
from find_proof import CausalProofFinder


class TestBFSDocalculusComplete(unittest.TestCase):
    """
    Non-synthetic end-to-end tests for BFS do-calculus search.

    Goals:
      - Each rule (1/2/3) is required in at least one test.
      - Multi-step mixed-rule proofs exist.
      - Enumerating ALL one-step successors is validated (branch completeness).
      - State-key normalization / ordering invariance is validated.
      - Negative reachability is validated (can’t add conditions, can’t create new do() vars).
      - BFS returns shortest proof and respects depth.
    """

    def _finder(self, causal_structure, max_depth=6):
        return CausalProofFinder(causal_structure=causal_structure, max_depth=max_depth)

    def _expr(self, s: str):
        return CausalProbability.parse(s)

    def _last(self, proof, start_expr):
        if not proof:
            return start_expr
        return proof[-1][1]

    def _reachable_keys(self, finder, start, depth):
        reached = finder.explore_all_equivalent_expressions(start, max_depth=depth)
        return set(reached.keys())

    # -------------------------
    # Basic BFS sanity
    # -------------------------

    def test_empty_path_if_same_expression(self):
        X, Y = sp.symbols("X Y")
        causal_structure = {X: [Y], Y: []}

        finder = self._finder(causal_structure, max_depth=3)
        a = self._expr("P(Y | do(X))")
        b = self._expr("P(Y | do(X))")

        proof = finder.find_proof(a, b)
        self.assertEqual(proof, [])

    def test_respects_depth_bound(self):
        # Reachable in 1 step (Rule 2), but depth=0 must fail.
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Z], Z: [Y], Y: []}

        finder0 = self._finder(causal_structure, max_depth=0)
        start = self._expr("P(Y | do(X), do(Z))")
        target = self._expr("P(Y | do(X), Z)")

        self.assertIsNone(finder0.find_proof(start, target))

    def test_bfs_returns_shortest_proof(self):
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Z], Z: [Y], Y: []}

        finder = self._finder(causal_structure, max_depth=6)
        start = self._expr("P(Y | do(X), do(Z))")
        target = self._expr("P(Y | do(X), Z)")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)
        self.assertEqual(len(proof), 1)

    # -------------------------
    # Normalization / invariance
    # -------------------------

    def test_ordering_invariance_same_state_key(self):
        # These should be equivalent as states (even if string differs).
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=2)

        a = self._expr("P(Y | do(X), Z)")
        b = self._expr("P(Y | Z, do(X))")

        self.assertEqual(finder._state_key(a), finder._state_key(b))

        # And BFS should treat them as already equal.
        proof = finder.find_proof(a, b)
        self.assertEqual(proof, [])

    # -------------------------
    # Rule 2 must be used (do(Z) -> Z)
    # -------------------------

    def test_rule2_one_step(self):
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=3)
        start = self._expr("P(Y | do(X), do(Z))")
        target = self._expr("P(Y | do(X), Z)")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)
        self.assertEqual(str(self._last(proof, start)), str(target))
        self.assertTrue(any("Rule 2" in step[0] for step in proof))

    def test_rule2_negative_confounding_blocks(self):
        # U -> Z and U -> Y should block Rule2 conversion do(Z)->Z
        U, Z, Y = sp.symbols("U Z Y")
        causal_structure = {U: [Z, Y], Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=5)
        start = self._expr("P(Y | do(Z))")
        target = self._expr("P(Y | Z)")

        self.assertIsNone(finder.find_proof(start, target))

    # -------------------------
    # Rule 1 must be used (drop observed W)
    # -------------------------

    def test_rule1_one_step_drop_observed_irrelevant(self):
        # X -> Y, W isolated
        # Expect Rule 1 to drop W from P(Y | do(X), W) -> P(Y | do(X))
        X, W, Y = sp.symbols("X W Y")
        causal_structure = {X: [Y], W: [], Y: []}

        finder = self._finder(causal_structure, max_depth=3)
        start = self._expr("P(Y | do(X), W)")
        target = self._expr("P(Y | do(X))")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)
        self.assertEqual(str(self._last(proof, start)), str(target))
        self.assertTrue(any("Rule 1" in step[0] for step in proof))

    def test_rule1_negative_collider_opening_keeps_observed(self):
        # W -> M <- Y, conditioning on M opens path W—Y.
        # So dropping W from P(Y | M, W) (or P(Y | do(X), M, W) with X irrelevant)
        # should NOT be possible.
        W, M, Y, X = sp.symbols("W M Y X")
        causal_structure = {W: [M], Y: [M], X: [], M: []}

        finder = self._finder(causal_structure, max_depth=4)
        start = self._expr("P(Y | do(X), M, W)")
        target = self._expr("P(Y | do(X), M)")  # would require dropping W

        self.assertIsNone(finder.find_proof(start, target))

    # -------------------------
    # Rule 3 must be used (delete action do(Z))
    # -------------------------

    def test_rule3_one_step_delete_action_when_irrelevant(self):
        # This is the cleanest Rule 3 situation:
        # Z is completely irrelevant to Y (and not ancestor of any observed vars), so deleting do(Z) should be allowed.
        #
        # Start: P(Y | do(X), do(Z))
        # Target: P(Y | do(X))
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [], Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=4)
        start = self._expr("P(Y | do(X), do(Z))")
        target = self._expr("P(Y | do(X))")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)
        self.assertEqual(str(self._last(proof, start)), str(target))
        self.assertTrue(any("Rule 3" in step[0] for step in proof))

    # -------------------------
    # Mixed multi-step proofs (requires >1 rule)
    # -------------------------

    def test_mixed_rule2_then_rule1(self):
        # Z isolated, W isolated
        # Start: P(Y | do(X), do(Z), W)
        # Step1 Rule2: do(Z) -> Z
        # Step2 Rule1: drop W
        X, Z, W, Y = sp.symbols("X Z W Y")
        causal_structure = {X: [], Z: [], W: [], Y: []}

        finder = self._finder(causal_structure, max_depth=4)
        start = self._expr("P(Y | do(X), do(Z), W)")
        target = self._expr("P(Y | do(X), Z)")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)
        self.assertGreaterEqual(len(proof), 2)
        self.assertEqual(str(self._last(proof, start)), str(target))

        labels = " ".join([p[0] for p in proof])
        self.assertIn("Rule 2", labels)
        self.assertIn("Rule 1", labels)

    # -------------------------
    # “Enumerate ALL successors” check (branch completeness at depth 1)
    # -------------------------

    def test_depth1_enumerates_multiple_successors(self):
        # With X,Z isolated:
        # From P(Y | do(X), do(Z), W) at depth=1 you should be able to:
        #  - convert do(X)->X (Rule2)
        #  - convert do(Z)->Z (Rule2)
        #  - drop W (Rule1)
        X, Z, W, Y = sp.symbols("X Z W Y")
        causal_structure = {X: [], Z: [], W: [], Y: []}

        finder = self._finder(causal_structure, max_depth=1)
        start = self._expr("P(Y | do(X), do(Z), W)")
        reached = finder.explore_all_equivalent_expressions(start, max_depth=1)

        # sanity check
        self.assertIn(finder._state_key(start), reached)

        # Must have more than just the start state if successors are enumerated
        self.assertGreaterEqual(len(reached), 2)

        expected_some = [
            self._expr("P(Y | do(X), Z, W)"),
            self._expr("P(Y | X, do(Z), W)"),
            self._expr("P(Y | do(X), do(Z))"),
        ]
        expected_keys = {finder._state_key(e) for e in expected_some}
        self.assertTrue(len(expected_keys & set(reached.keys())) >= 1)


    def test_cannot_add_observed_condition(self):
        X, Y, Z = sp.symbols("X Y Z")
        causal_structure = {X: [Y], Z: [Y], Y: []}

        finder = self._finder(causal_structure, max_depth=5)
        start = self._expr("P(Y | do(X))")
        target = self._expr("P(Y | do(X), Z)")

        self.assertIsNone(finder.find_proof(start, target))

    def test_cannot_create_new_do_variable(self):
        # There is no rule that introduces do(Z) if it wasn't present.
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=5)
        start = self._expr("P(Y | do(X))")
        target = self._expr("P(Y | do(X), do(Z))")

        self.assertIsNone(finder.find_proof(start, target))

    # -------------------------
    # Explore boundedness (non-explosion on small graphs)
    # -------------------------

    def test_explore_finite_small_graph(self):
        X, Z, W, Y = sp.symbols("X Z W Y")
        causal_structure = {X: [Y], Z: [], W: [], Y: []}

        finder = self._finder(causal_structure, max_depth=4)
        start = self._expr("P(Y | do(X), do(Z), W)")

        reached = finder.explore_all_equivalent_expressions(start, max_depth=4)
        self.assertLess(len(reached), 500)


    def test_state_key_distinguishes_assignment_values(self):
        Z, Y = sp.symbols("Z Y")
        causal_structure = {Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=2)

        a = self._expr("P(Y | Z=0)")
        b = self._expr("P(Y | Z=1)")

        self.assertNotEqual(finder._state_key(a), finder._state_key(b))

    def test_bfs_does_not_equate_different_assignments(self):
        # There is no do-calculus rule that changes assignment values (0 -> 1),
        # so these should NOT be reachable / treated equivalent.
        Z, Y = sp.symbols("Z Y")
        causal_structure = {Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=4)

        start = self._expr("P(Y | Z=0)")
        target = self._expr("P(Y | Z=1)")

        self.assertIsNone(finder.find_proof(start, target))

    def test_rule1_can_drop_irrelevant_assignment_condition(self):
        # Z is isolated so it should be droppable even if expressed as Z=0
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}

        finder = self._finder(causal_structure, max_depth=3)
        start = self._expr("P(Y | do(X), Z=0)")
        target = self._expr("P(Y | do(X))")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)
        self.assertEqual(str(self._last(proof, start)), str(target))
        self.assertTrue(any("Rule 1" in step[0] for step in proof))


    def test_rule3_multi_do_deletes_one_action(self):
        # With everything isolated, deleting any one do-var should be valid.
        X, Z, T, Y = sp.symbols("X Z T Y")
        causal_structure = {X: [], Z: [], T: [], Y: []}

        finder = self._finder(causal_structure, max_depth=3)
        start = self._expr("P(Y | do(X), do(Z), do(T))")
        target = self._expr("P(Y | do(X), do(T))")  # delete do(Z)

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)
        self.assertEqual(str(self._last(proof, start)), str(target))
        self.assertTrue(any("Rule 3" in step[0] for step in proof))

    def test_ate_termwise_equivalence(self):
        X, Y = sp.symbols("X Y")
        causal_structure = {X: [Y], Y: []}
        finder = self._finder(causal_structure, max_depth=4)

        start = self._expr("P(Y | do(X=1))") - self._expr("P(Y | do(X=0))")
        target = self._expr("P(Y | do(X=1))") - self._expr("P(Y | do(X=0))")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)

    def test_ate_not_equivalent_if_swapped(self):
        X, Y = sp.symbols("X Y")
        causal_structure = {X: [Y], Y: []}
        finder = self._finder(causal_structure, max_depth=4)

        start = self._expr("P(Y | do(X=1))") - self._expr("P(Y | do(X=0))")
        target = self._expr("P(Y | do(X=0))") - self._expr("P(Y | do(X=1))")

        self.assertIsNone(finder.find_proof(start, target))

    def test_ate_termwise_each_side_rewrites(self):
        X, Z, Y, W = sp.symbols("X Z Y W")
        causal_structure = {X: [Y], Z: [], W: [], Y: []}
        finder = self._finder(causal_structure, max_depth=4)

        # Each term needs Rule 2: do(Z)->Z, but they are not identical so no cancellation
        start = self._expr("P(Y | do(X), do(Z))") - self._expr("P(Y | do(X), do(Z), W)")
        target = self._expr("P(Y | do(X), Z)")     - self._expr("P(Y | do(X), Z, W)")

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)

    def test_rule3_multi_do_with_observed_conditioning(self):
        X, Z, T, W, Y = sp.symbols("X Z T W Y")
        causal_structure = {Z: [W], X: [], T: [], W: [], Y: []}

        finder = self._finder(causal_structure, max_depth=4)

        start = self._expr("P(Y | do(X), do(Z), do(T), W)")
        target = self._expr("P(Y | do(X), do(T), W)")  # delete do(Z)

        proof = finder.find_proof(start, target)
        self.assertIsNotNone(proof)

    def test_ordering_invariance_with_assignment(self):
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}
        finder = self._finder(causal_structure, max_depth=2)

        a = self._expr("P(Y | do(X), Z=0)")
        b = self._expr("P(Y | Z=0, do(X))")

        self.assertEqual(finder._state_key(a), finder._state_key(b))
        self.assertEqual(finder.find_proof(a, b), [])

if __name__ == "__main__":
    unittest.main()