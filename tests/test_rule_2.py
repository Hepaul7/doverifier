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


class TestRule2All(unittest.TestCase):
    """
    Rule 2 enumerator should return ALL one-step conversions do(Z)->Z
    that satisfy the Rule 2 d-sep criterion.
    """

    def test_returns_all_convertible_do_vars(self):
        # X -> Y, Z isolated, T isolated
        X, Y, Z, T = sp.symbols("X Y Z T")
        causal_structure = {X: [Y], Z: [], T: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(X), do(Z), do(T))")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_2_all()
        outs_str = {str(o) for o in outs}

        expect_convert_z = str(CausalProbability.parse("P(Y | do(X), Z, do(T))"))
        expect_convert_t = str(CausalProbability.parse("P(Y | do(X), do(Z), T)"))

        self.assertTrue(expect_convert_z in outs_str or
                        str(CausalProbability.parse("P(Y | Z, do(X), do(T))")) in outs_str)
        self.assertTrue(expect_convert_t in outs_str or
                        str(CausalProbability.parse("P(Y | T, do(X), do(Z))")) in outs_str)

        # Soundness: each successor reduces do-count by exactly 1
        before_do = len(_do_vars(expr))
        for o in outs:
            self.assertEqual(len(_do_vars(o)), before_do - 1)

    def test_does_not_convert_under_confounding(self):
        # U -> Z and U -> Y => confounding; should not convert do(Z)
        U, Z, Y = sp.symbols("U Z Y")
        causal_structure = {U: [Z, Y], Z: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(Z))")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_2_all()
        self.assertEqual(outs, [])

    def test_eq_observed_does_not_crash(self):
        X, Z, Y = sp.symbols("X Z Y")
        causal_structure = {X: [Y], Z: [], Y: []}

        expr = CausalProbability.parse("P(Y | do(Z), X=0)")
        ce = CausalExpr(expr, causal_structure)

        outs = ce.apply_rule_2_all()
        self.assertTrue(isinstance(outs, list))


if __name__ == "__main__":
    unittest.main()