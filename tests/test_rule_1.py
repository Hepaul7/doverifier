import unittest
import sympy as sp

from probability import CausalProbability, Do
from causal_equiv import CausalExpr


def _conds(expr):
    return list(expr.args[1:]) if hasattr(expr, "args") and len(expr.args) > 1 else []


def _do_vars(expr):
    return {c.args[0] for c in _conds(expr) if isinstance(c, Do)}


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


class TestRule1All(unittest.TestCase):
    """
    Rule 1 enumerator should return ALL one-step drops of observed vars W
    that satisfy the Rule 1 d-sep criterion.
    """

    def test_returns_all_droppable_observed_vars(self):
        # X -> Y, and U, V isolated observed vars.
        # From P(Y | do(X), U, V), BOTH U and V are droppable in one step.
        X, Y, U, V = sp.symbols("X Y U V")
        causal_structure = {X: [Y], U: [], V: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), U, V)")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_1_all()
        outs_str = {str(o) for o in outs}

        expected_drop_u = str(CausalProbability.parse("P(Y | do(X), V)"))
        expected_drop_v = str(CausalProbability.parse("P(Y | do(X), U)"))

        self.assertEqual(outs_str, {expected_drop_u, expected_drop_v})

        for o in outs:
            self.assertEqual(_do_vars(o), {X})
            self.assertEqual(len(_obs_vars(expr) - _obs_vars(o)), 1)

    def test_returns_empty_when_no_observed_vars(self):
        X, Y = sp.symbols("X Y")
        causal_structure = {X: [Y], Y: []}

        expr = CausalProbability.parse("P(Y | do(X))")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_1_all()
        self.assertEqual(outs, [])

    def test_does_not_drop_relevant_observed(self):
        # W -> Y; W is relevant, so should not be droppable
        X, Y, W = sp.symbols("X Y W")
        causal_structure = {W: [Y], X: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), W)")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_1_all()
        self.assertEqual(outs, [])

    def test_eq_observed_does_not_crash_and_can_drop(self):
        # U isolated, observed as U=0; should be droppable like observing U
        X, Y, U = sp.symbols("X Y U")
        causal_structure = {X: [Y], U: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), U=0)")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_1_all()
        outs_str = {str(o) for o in outs}
        self.assertIn(str(CausalProbability.parse("P(Y | do(X))")), outs_str)


if __name__ == "__main__":
    unittest.main()