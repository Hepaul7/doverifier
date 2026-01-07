import logging
from collections import deque
import networkx as nx
import sympy as sp

from probability import CausalProbability, Do
from causal_equiv import CausalExpr

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CausalProofFinder:
    """
    Proof search over do-calculus only (Rules 1/2/3), using BFS.

    Completeness requirement:
      - We must enumerate ALL valid one-step rewrites from each state.
      - Therefore we use apply_rule_{1,2,3}_all() and branch on all successors.
    """

    def __init__(self, causal_structure=None, max_depth=10):
        self.causal_structure = causal_structure
        self.max_depth = max_depth
        self._validate_causal_structure()

    def _validate_causal_structure(self):
        """Validate that the causal structure is acyclic (DAG). If cycles exist, try to break them."""
        if not self.causal_structure:
            return

        G = nx.DiGraph()
        for parent, children in self.causal_structure.items():
            G.add_node(str(parent))
            for child in children:
                G.add_node(str(child))
                G.add_edge(str(parent), str(child))

        if nx.is_directed_acyclic_graph(G):
            return

        logger.warning("Causal structure contains cycles! Attempting to break cycles.")
        try:
            cycles = list(nx.simple_cycles(G))
            logger.warning(f"Cycles detected: {cycles}")

            G_fixed = G.copy()
            for cycle in cycles:
                # remove one edge per cycle (simple heuristic)
                edge = (cycle[-1], cycle[0])
                if G_fixed.has_edge(*edge):
                    G_fixed.remove_edge(*edge)
                    logger.info(f"Removed edge {edge} to break cycle")

            # Rebuild structure in sympy symbols
            new_structure = {}
            for node in G_fixed.nodes():
                children = list(G_fixed.successors(node))
                if children:
                    node_sym = sp.Symbol(node)
                    children_sym = [sp.Symbol(c) for c in children]
                    new_structure[node_sym] = children_sym

            self.causal_structure = new_structure
            logger.info("Causal structure fixed to an acyclic version.")
        except Exception as e:
            logger.error(f"Error fixing cycles: {e}")

    def _are_expressions_equivalent(self, expr1, expr2):
        """
        Structural equivalence for *single* CausalProbability expressions:
        - Same outcome
        - Same set of Do(...) conditions
        - Same set of observed conditions
        """
        if not isinstance(expr1, CausalProbability) or not isinstance(expr2, CausalProbability):
            return False

        if expr1.args[0] != expr2.args[0]:
            return False

        def split(expr):
            do_ops = set()
            obs = set()
            for cond in expr.args[1:]:
                if isinstance(cond, Do):
                    do_ops.add(cond)
                else:
                    obs.add(cond)
            return do_ops, obs

        d1, o1 = split(expr1)
        d2, o2 = split(expr2)
        return d1 == d2 and o1 == o2

    def _as_subtraction_pair(self, expr):
        """
        If expr is of the form A - B, return (A, B).
        SymPy represents subtraction as Add(A, -B).
        """
        expr = sp.simplify(expr)
        if not isinstance(expr, sp.Add):
            return None

        pos = []
        neg = []
        for t in expr.as_ordered_terms():
            coeff, rest = t.as_coeff_Mul()
            if coeff == 1:
                pos.append(rest)
            elif coeff == -1:
                neg.append(rest)
            else:
                return None

        if len(pos) == 1 and len(neg) == 1:
            return pos[0], neg[0]
        return None

    def _state_key(self, expr):
        if not isinstance(expr, CausalProbability):
            return str(expr).replace(" ", "")

        outcome = expr.args[0]

        do_vars = []
        obs_vars = []

        for cond in expr.args[1:]:
            if isinstance(cond, Do):
                do_vars.append(str(cond))
            else:
                obs_vars.append(str(cond))  # keep Eq(Z,0) distinct

        do_vars.sort()
        obs_vars.sort()

        return f"Y={outcome}|DO={','.join(do_vars)}|OBS={','.join(obs_vars)}"

    def _do_calculus_successors(self, expr):
        """
        Return list of (rule_label, next_expr) for ALL one-step do-calculus rewrites.
        """
        ce = CausalExpr(expr, self.causal_structure)

        successors = []

        # Rule ordering only affects search order, NOT completeness
        rule_gens = [
            (1, ce.apply_rule_1_all),
            (2, ce.apply_rule_2_all),
            (3, ce.apply_rule_3_all),
        ]

        for rule_num, gen in rule_gens:
            try:
                outs = gen()  # list of expressions
            except Exception as e:
                logger.debug(f"Rule {rule_num} enumeration failed on {expr}: {e}")
                continue

            for out in outs:
                if self._are_expressions_equivalent(out, expr):
                    continue
                successors.append((f"Do-calculus Rule {rule_num}", out))

        uniq = []
        seen = set()
        for label, out in successors:
            k = self._state_key(out)
            if k in seen:
                continue
            seen.add(k)
            uniq.append((label, out))
        return uniq


    def find_proof(self, start_expr, target_expr):
        """
        Find a do-calculus-only proof path from start_expr to target_expr.
        Returns:
          - [] if already equivalent
          - list[(rule_label, expr)] if found
          - None if not found within max_depth
        """
        # Case 1: single probability expressions (existing behavior)
        if isinstance(start_expr, CausalProbability) and isinstance(target_expr, CausalProbability):
            return self._find_proof_single(start_expr, target_expr)

        # Case 2: ATE-style subtraction A - B
        spair = self._as_subtraction_pair(start_expr)
        tpair = self._as_subtraction_pair(target_expr)

        if spair is not None and tpair is not None:
            A1, B1 = spair
            A2, B2 = tpair

            if not all(isinstance(x, CausalProbability) for x in (A1, B1, A2, B2)):
                raise TypeError("ATE terms must be CausalProbability instances")

            # Prove left term and right term separately (order matters for subtraction)
            p_left = self._find_proof_single(A1, A2)
            if p_left is None:
                return None
            p_right = self._find_proof_single(B1, B2)
            if p_right is None:
                return None

            # Return a structured proof (keeps backward compatibility for single case)
            return [("ATE-left", p_left), ("ATE-right", p_right)]
        raise TypeError("Unsupported expression type: expected CausalProbability or A-B of CausalProbability")

    def _find_proof_single(self, start_expr, target_expr):
        if self._are_expressions_equivalent(start_expr, target_expr):
            logger.info("Expressions are already equivalent")
            return []

        start_key = self._state_key(start_expr)
        target_key = self._state_key(target_expr)

        queue = deque([(start_expr, [])])
        visited = {start_key}

        while queue:
            cur, path = queue.popleft()

            if len(path) >= self.max_depth:
                continue

            # Goal check (structural, then fallback to key)
            if self._are_expressions_equivalent(cur, target_expr) or self._state_key(cur) == target_key:
                logger.info(f"Found proof with {len(path)} steps")
                return path

            for rule_label, nxt in self._do_calculus_successors(cur):
                k = self._state_key(nxt)
                if k in visited:
                    continue
                visited.add(k)

                new_path = path + [(rule_label, nxt)]

                if self._are_expressions_equivalent(nxt, target_expr) or k == target_key:
                    logger.info(f"Found proof with {len(new_path)} steps")
                    return new_path

                queue.append((nxt, new_path))

        logger.info(f"No proof found within {self.max_depth} steps")
        return None

    def explore_all_equivalent_expressions(self, start_expr, max_depth=None):
        """
        Explore all expressions reachable by do-calculus (rules 1/2/3) within max_depth.
        Returns dict: key_str -> (expr_obj, last_rule_label)
        """
        if max_depth is None:
            max_depth = self.max_depth

        if not isinstance(start_expr, CausalProbability):
            raise TypeError("Start expression must be a CausalProbability instance")

        start_key = self._state_key(start_expr)

        queue = deque([(start_expr, 0)])
        visited = {start_key}
        all_reached = {start_key: (start_expr, "Initial")}

        while queue:
            cur, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for rule_label, nxt in self._do_calculus_successors(cur):
                k = self._state_key(nxt)
                if k in visited:
                    continue
                visited.add(k)
                all_reached[k] = (nxt, rule_label)
                queue.append((nxt, depth + 1))

        logger.info(f"Found {len(all_reached)} reachable expressions within {max_depth} steps")
        return all_reached


