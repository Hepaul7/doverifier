import unittest
import sympy as sp

from probability import CausalProbability, Do, Mult 
from probability import SumOver


class TestProbability(unittest.TestCase):

    def test_parse_marginal(self):
        expr = CausalProbability.parse("P(Y)")
        self.assertIsInstance(expr, CausalProbability)
        self.assertEqual(str(expr), "P(Y)")

    def test_parse_conditional_symbol(self):
        expr = CausalProbability.parse("P(Y|X)")
        self.assertIsInstance(expr, CausalProbability)
        self.assertEqual(str(expr), "P(Y | X)")

    def test_parse_assignments(self):
        expr = CausalProbability.parse("P(Y=1|X=0)")
        self.assertIsInstance(expr, CausalProbability)
        self.assertIn("Y=1", str(expr))
        self.assertIn("X=0", str(expr))

    def test_parse_do_no_value(self):
        expr = CausalProbability.parse("P(Y|do(X))")
        self.assertIsInstance(expr, CausalProbability)
        # do(X) should appear
        self.assertIn("do(X)", str(expr))

    def test_parse_do_with_value(self):
        expr = CausalProbability.parse("P(Y=1|do(X=0))")
        self.assertIsInstance(expr, CausalProbability)
        self.assertIn("Y=1", str(expr))
        self.assertIn("do(X=0)", str(expr))

    def test_parse_do_multiple_vars(self):
        expr = CausalProbability.parse("P(Y|do(X, V2))")
        self.assertIsInstance(expr, CausalProbability)
        s = str(expr)
        self.assertIn("do(X)", s)
        self.assertIn("do(V2)", s)

    def test_parse_subscript_do(self):
        expr = CausalProbability.parse("P(Y_{X=1, V2=0})")
        self.assertIsInstance(expr, CausalProbability)
        s = str(expr)
        self.assertIn("P(Y", s)
        self.assertIn("do(X=1)", s)
        self.assertIn("do(V2=0)", s)

    def test_order_insensitive_conditions(self):
        a = CausalProbability.parse("P(Y | do(X), V2)")
        b = CausalProbability.parse("P(Y | V2, do(X))")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(str(a), str(b))

    def test_order_insensitive_multiple(self):
        a = CausalProbability.parse("P(Y | do(X), do(V2), Z)")
        b = CausalProbability.parse("P(Y | Z, do(V2), do(X))")
        self.assertEqual(a, b)
        self.assertEqual(str(a), str(b))

    def test_hashable_content_stable(self):
        a = CausalProbability.parse("P(Y | do(X), Z)")
        # should be usable as dict key
        d = {a: 1}
        b = CausalProbability.parse("P(Y | Z, do(X))")
        self.assertEqual(d[b], 1)

    def test_product_parsing_mult(self):
        expr = CausalProbability.parse("P(A|B) * P(B)")
        self.assertTrue(isinstance(expr, (Mult, sp.Mul)))       
        self.assertEqual(len(expr.args), 2)
        self.assertTrue(all(isinstance(arg, CausalProbability) for arg in expr.args))

    def test_arithmetic_parsing_difference(self):
        expr = CausalProbability.parse("P(Y_{X=1}) - P(Y_{X=0})")
        self.assertIsInstance(expr, sp.Expr)
        s = str(expr)
        self.assertIn("P(Y", s)
        self.assertIn("do(X=1)", s)
        self.assertIn("do(X=0)", s)

    def test_arithmetic_parsing_mixture(self):
        expr = CausalProbability.parse("P(Y_{X=1}) - P(Y_{X=0}) + P(Z)")
        self.assertIsInstance(expr, sp.Expr)
        s = str(expr)
        self.assertIn("P(Z)", s)

    def test_do_object_string(self):
        X = sp.Symbol("X")
        self.assertEqual(str(Do(X)), "do(X)")
        self.assertEqual(str(Do(X, 0)), "do(X=0)")

    def test_sumover_basic(self):
        Y, X, Z = sp.symbols("Y X Z")
        inner = CausalProbability(Y, X, Z) * CausalProbability(Z)
        expr = SumOver(Z, inner)
        self.assertIn("Σ", str(expr))
        self.assertIn("Z", str(expr))
    
    def test_sumover_composes(self):
        Y, X, Z = sp.symbols("Y X Z")
        inner = CausalProbability(Y, X, Z) * CausalProbability(Z)
        expr = SumOver(Z, inner)
        combined = expr + expr
        self.assertEqual(sp.simplify(combined - 2*expr), 0)


if __name__ == "__main__":
    unittest.main()