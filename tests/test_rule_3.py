import unittest
import sympy as sp

from probability import CausalProbability, Do
from causal_equiv import CausalExpr


def _conds(expr):
    return list(expr.args[1:]) if hasattr(expr, "args") and len(expr.args) > 1 else []


def _do_vars(expr):
    return {c.args[0] for c in _conds(expr) if isinstance(c, Do)}


def _do_count(expr):
    return sum(isinstance(c, Do) for c in _conds(expr))


def _obs_vars(expr):
    obs = set()
    for c in _conds(expr):
        if isinstance(c, Do):
            continue
        if isinstance(c, sp.Equality):
            obs.add(c.lhs)
        else:
            obs |= set(getattr(c, "free_symbols", {c}))
    return obs


class TestRule3All(unittest.TestCase):
    """
    Rule 3 enumerator should return ALL one-step deletions of some do(Z)
    holding some other do(X) fixed (ordered pairs).
    """

    def test_returns_both_deletions_when_both_irrelevant(self):
        # Y isolated; X and Z both isolated interventions.
        # From P(Y | do(X), do(Z)) we should be able to delete do(X) (keeping Z)
        # AND delete do(Z) (keeping X), so 2 successors.
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [], Z: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), do(Z))")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_3_all()
        outs_str = {str(o) for o in outs}

        exp_keep_x = str(CausalProbability.parse("P(Y | do(X))"))
        exp_keep_z = str(CausalProbability.parse("P(Y | do(Z))"))

        self.assertEqual(outs_str, {exp_keep_x, exp_keep_z})

        # Soundness: each successor removes exactly one do
        for o in outs:
            self.assertEqual(_do_count(o), _do_count(expr) - 1)

    def test_only_deletes_auxiliary_when_X_affects_Y(self):
        # X -> Y, Z isolated
        # Deleting do(Z) while keeping do(X) should be possible.
        # Deleting do(X) while keeping do(Z) should NOT be possible (X still affects Y).
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), do(Z))")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_3_all()
        outs_str = {str(o) for o in outs}

        expect = str(CausalProbability.parse("P(Y | do(X))"))
        self.assertIn(expect, outs_str)

        # Ensure we did NOT delete do(X)
        self.assertNotIn(str(CausalProbability.parse("P(Y | do(Z))")), outs_str)

    def test_does_not_delete_when_Z_affects_Y(self):
        # Z -> Y, plus do(X) present.
        # We must not be able to delete do(Z) (keeping do(X)).
        # (We might delete do(X) if X irrelevant — that's fine.)
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {Z: [Y], X: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), do(Z))")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_3_all()
        # No successor should drop do(Z) and leave do(X) only
        self.assertNotIn(str(CausalProbability.parse("P(Y | do(X))")), {str(o) for o in outs})
        # But ensure do(Z) is not deleted in any successor where the remaining do-set excludes Z
        for o in outs:
            self.assertIn(Z, _do_vars(o))

    def test_ancestor_of_W_guard(self):
        # U -> Z and U -> Y (confounding), and Z -> W, and condition on W.
        # The "ancestor of W" logic should prevent removing incoming to Z, so Z remains tied to Y via U.
        # Therefore deletion of do(Z) (keeping do(X)) should NOT be allowed.
        U, X, Z, W, Y = sp.symbols("U X Z W Y")
        causal_structure = {U: [Z, Y], Z: [W], X: [], W: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), do(Z), W)")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_3_all()
        outs_str = {str(o) for o in outs}

        # specifically, we should NOT be able to delete do(Z) leaving do(X) (and W)
        self.assertNotIn(str(CausalProbability.parse("P(Y | do(X), W)")), outs_str)

    def test_eq_observed_does_not_crash(self):
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), do(Z), X=0)")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_3_all()
        self.assertTrue(isinstance(outs, list))

    def test_rule3_conditions_on_all_kept_do_and_observed(self):
        # Z is irrelevant to Y only when conditioning on BOTH do(X) and W.
        X, Z, W, Y = sp.symbols("X Z W Y")
        causal_structure = {
            X: [Y],
            Z: [W],
            W: [],
            Y: []
        }

        # Conditioning on W blocks Z -> W -> ...
        expr = CausalProbability.parse("P(Y | do(X), do(Z), W)")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_3_all()
        outs_str = {str(o) for o in outs}

        # Deleting do(Z) should be allowed ONLY because W is conditioned
        self.assertIn(str(CausalProbability.parse("P(Y | do(X), W)")), outs_str)

    def test_rule3_do_order_invariance(self):
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [], Z: [], Y: []}

        expr1 = CausalProbability.parse("P(Y | do(X), do(Z))")
        expr2 = CausalProbability.parse("P(Y | do(Z), do(X))")

        ce1 = CausalExpr(expr1, causal_structure)
        ce2 = CausalExpr(expr2, causal_structure)

        outs1 = {str(o) for o in ce1.apply_rule_3_all()}
        outs2 = {str(o) for o in ce2.apply_rule_3_all()}

        self.assertEqual(outs1, outs2)

    def test_rule3_removes_exactly_one_do(self):
        X, Z, T, Y = sp.symbols("X Z T Y")
        causal_structure = {X: [], Z: [], T: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), do(Z), do(T))")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_3_all()

        for o in outs:
            self.assertEqual(_do_count(o), _do_count(expr) - 1)

if __name__ == "__main__":
    unittest.main()