import networkx as nx
import sympy as sp
from collections import deque
import logging

logger = logging.getLogger(__name__)

import networkx as nx

def is_d_separated(graph: nx.DiGraph, start, end, conditioned_set):
    """
    Returns True iff start ⟂ end | conditioned_set in the DAG 'graph'
    using: ancestral subgraph + moralization + remove Z + undirected separation.
    """
    Z = set(conditioned_set)

    if start == end:
        return False
    if start not in graph or end not in graph:
        return True

    nodes_of_interest = {start, end} | Z

    anc = set(nodes_of_interest)
    for n in nodes_of_interest:
        anc |= nx.ancestors(graph, n)

    G_anc = graph.subgraph(anc).copy()

    moral = nx.Graph()
    moral.add_nodes_from(G_anc.nodes())
    moral.add_edges_from(G_anc.edges())  # skeleton

    for child in G_anc.nodes():
        parents = list(G_anc.predecessors(child))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                moral.add_edge(parents[i], parents[j])

    moral.remove_nodes_from([z for z in Z if z in moral])

    if start not in moral or end not in moral:
        return True

    return not nx.has_path(moral, start, end)

def has_descendant_in_set(graph, node, conditioned_set, visited=None):
    if visited is None:
        visited = set()
    if node in visited:
        return False
    visited.add(node)
    for child in graph.successors(node):
        if child in conditioned_set:
            return True
        if has_descendant_in_set(graph, child, conditioned_set, visited):
            return True
    return False



class CausalExpr:
    """
    Represents a causal expression with associated causal graph.
    """
    def __init__(self, expression, causal_structure=None):
        self.expression = expression
        self.causal_structure = causal_structure
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """
        Build a directed graph from the causal structure.
        Ensures the graph is a DAG by checking for cycles.
        """
        G = nx.DiGraph()
        
        if self.causal_structure:
            for parent in self.causal_structure:
                G.add_node(str(parent))
                for child in self.causal_structure[parent]:
                    G.add_node(str(child))
            
            for parent, children in self.causal_structure.items():
                for child in children:
                    G.add_edge(str(parent), str(child))
            
            try:
                cycles = list(nx.simple_cycles(G))
                if cycles:
                    logger.warning(f"Warning: The graph contains cycles: {cycles}")
            except Exception as e:
                logger.error(f"Error checking for cycles: {e}")
        
        return G
    
    def copy(self):
        """Create a deep copy of this expression."""
        return CausalExpr(self.expression.copy() if hasattr(self.expression, 'copy') else self.expression, 
                         self.causal_structure)
    
    def _make_acyclic_copy(self, graph=None):
        """
        Make an acyclic copy of the graph, removing edges to break cycles if necessary.
        This ensures we can perform d-separation checks.
        """
        if graph is None:
            graph = self.graph
            
        g_copy = graph.copy()
        
        max_iterations = 100  
        iteration = 0
        
        while iteration < max_iterations:
            if nx.is_directed_acyclic_graph(g_copy):
                break
                
            try:
                cycle = nx.find_cycle(g_copy)
                if not cycle:
                    break
                g_copy.remove_edge(*cycle[-1])
                iteration += 1
            except nx.NetworkXNoCycle:
                break
            except Exception as e:
                logger.error(f"Error making graph acyclic: {e}")
                break
        
        if iteration >= max_iterations:
            logger.warning("Reached maximum iterations trying to make graph acyclic")
            
        return g_copy
    
    def _custom_d_separation(self, X, Y, Z, graph=None):
        if graph is None:
            graph = self.graph
        return is_d_separated(graph, str(X), str(Y), set(str(z) for z in Z))

        
    def _get_moral_graph(self, graph):
        """
        Create a moral graph from a DAG (connect parents of common children).
        This is a key step in checking d-separation.
        """
        moral = nx.Graph()
        
        for node in graph.nodes():
            moral.add_node(node)
        
        for u, v in graph.edges():
            moral.add_edge(u, v)
        
        for node in graph.nodes():
            parents = list(graph.predecessors(node))
            for i in range(len(parents)):
                for j in range(i+1, len(parents)):
                    moral.add_edge(parents[i], parents[j])
        
        return moral
    
    def _is_d_separated(self, X, Y, Z):
        """
        Check if X and Y are d-separated by Z in the causal graph.
        Returns True if they are d-separated, False otherwise.
        """
        try:
            X_str = str(X)
            Y_str = str(Y)
            Z_strs = [str(z) for z in Z]
            
            return self._custom_d_separation(X_str, Y_str, set(Z_strs))
        except Exception as e:
            logger.error(f"Error in d-separation check: {e}")
            return False
        
    def _create_intervention_graph(self, do_vars):
        """
        Create a modified graph where incoming edges to intervention variables are removed.
        This is G_X for do(X).
        """
        g_modified = self.graph.copy()
        
        for X in do_vars:
            incoming_edges = [(u, v) for u, v in g_modified.edges() if v == str(X)]
            g_modified.remove_edges_from(incoming_edges)
        
        return g_modified
    
    def apply_rule_1_all(self):
        """
        Enumerate all one-step Rule 1 rewrites:

        P(Y | do(X), Z, W) = P(Y | do(X), Z)
        if Y ⟂ W | X, Z in G_{\bar X}

        Returns a list of all distinct expressions obtainable by dropping
        exactly one observed condition W that satisfies the criterion.
        """
        from probability import CausalProbability, Do
        import sympy as sp

        outs = []
        seen = set()

        if not hasattr(self.expression, "args"):
            return outs

        Y = self.expression.args[0]
        conditions = list(self.expression.args[1:])
        if not conditions:
            return outs

        do_vars = []
        obs_vars = []

        for cond in conditions:
            if isinstance(cond, Do):
                do_vars.append(cond.args[0])
            else:
                if isinstance(cond, sp.Equality):
                    obs_vars.append(cond.lhs)
                else:
                    obs_vars.append(cond)

        if not obs_vars:
            return outs

        g_x = self._create_intervention_graph(do_vars)

        for W in obs_vars:
            other_obs = [v for v in obs_vars if v != W]
            conditioning_set = do_vars + other_obs  

            try:
                if self._custom_d_separation(Y, W, conditioning_set, g_x):
                    new_conditions = []
                    for c in conditions:
                        # drop the specific observed W (by object equality)
                        if c == W:
                            continue
                        # if original condition was Eq(X,0), we normalized it to X for obs_vars;
                        # don’t accidentally drop Eq(...) unless it corresponds to W.lhs == X
                        if isinstance(c, sp.Equality) and W in c.free_symbols:
                            # If W is that variable, then dropping "observing W" should drop this Eq condition.
                            if c.lhs == W:
                                continue
                        new_conditions.append(c)

                    cand = CausalProbability(Y, *new_conditions)
                    s = str(cand)
                    if s not in seen:
                        seen.add(s)
                        outs.append(cand)
            except Exception:
                # keep enumerating other candidates
                continue

        return outs
    
    def apply_rule_2_all(self):
        from probability import CausalProbability, Do
        import sympy as sp
        outs = []
        seen = set()

        if not hasattr(self.expression, "args"):
            return outs

        Y = self.expression.args[0]
        conditions = list(self.expression.args[1:])

        do_indices = []
        obs_vars = []
        for i, cond in enumerate(conditions):
            if isinstance(cond, Do):
                do_indices.append(i)
            else:
                # normalize Eq(X,0) -> X for graph conditioning safety
                if isinstance(cond, sp.Equality):
                    obs_vars.append(cond.lhs)
                else:
                    obs_vars.append(cond)

        if not do_indices:
            return outs

        # Try converting each do(Z) independently
        for idx in do_indices:
            Z = conditions[idx].args[0]
            other_do_vars = [conditions[j].args[0] for j in do_indices if j != idx]

            g_xz = self.graph.copy()

            # bar other do-vars: remove incoming edges to them
            for do_var in other_do_vars:
                incoming = [(u, v) for (u, v) in g_xz.edges() if v == str(do_var)]
                g_xz.remove_edges_from(incoming)

            # underline Z: remove outgoing edges from Z
            outgoing = [(u, v) for (u, v) in g_xz.edges() if u == str(Z)]
            g_xz.remove_edges_from(outgoing)

            try:
                conditioning_set = other_do_vars + obs_vars
                if self._custom_d_separation(Y, Z, conditioning_set, g_xz):
                    new_conditions = conditions.copy()
                    new_conditions[idx] = Z
                    cand = CausalProbability(Y, *new_conditions)
                    s = str(cand)
                    if s not in seen:
                        seen.add(s)
                        outs.append(cand)
            except Exception:
                pass

        return outs
    
    def apply_rule_3_all(self):
        """
        Rule 3 (Action/Intervention deletion), enumerated:

        P(Y | do(X), do(Z), W) = P(Y | do(X), W)
        if Y ⟂ Z | X, W in G_{bar X, bar Z(W)}
        where Z(W) are those Z that are NOT ancestors of any W in G_{bar X}
        (generalized to multiple do-vars: keep all other do-vars active).

        We enumerate removing each do-var Z while keeping the rest.
        """
        from probability import CausalProbability, Do
        import sympy as sp
        import networkx as nx

        outs = []
        seen = set()

        if not hasattr(self.expression, "args"):
            return outs

        Y = self.expression.args[0]
        conditions = list(self.expression.args[1:])

        # Identify do-vars (by index) and observed vars (variables only for d-sep)
        do_indices = []
        obs_vars = []
        for i, cond in enumerate(conditions):
            if isinstance(cond, Do):
                do_indices.append(i)
            else:
                if isinstance(cond, sp.Equality):
                    obs_vars.append(cond.lhs)
                else:
                    obs_vars.append(cond)

        if len(do_indices) == 0:
            return outs

        # Try removing each do(Z) in turn
        for z_idx in do_indices:
            Z = conditions[z_idx].args[0]

            # Kept interventions are all other do-vars
            kept_do_vars = [conditions[i].args[0] for i in do_indices if i != z_idx]

            # Build G_{bar kept_do_vars}: remove incoming edges to ALL kept do-vars
            g_bar = self.graph.copy()
            for V in kept_do_vars:
                incoming = [(u, v) for (u, v) in g_bar.edges() if v == str(V)]
                g_bar.remove_edges_from(incoming)

            # Determine whether Z is an ancestor of any observed W in this barred graph
            is_ancestor_of_obs = False
            for W in obs_vars:
                try:
                    if nx.has_path(g_bar, str(Z), str(W)):
                        is_ancestor_of_obs = True
                        break
                except nx.NetworkXError:
                    continue

            # If Z is NOT an ancestor of any observed var, we may also bar Z (remove incoming edges to Z)
            g_test = g_bar.copy()
            if not is_ancestor_of_obs:
                z_incoming = [(u, v) for (u, v) in g_test.edges() if v == str(Z)]
                g_test.remove_edges_from(z_incoming)

            # Independence test: Y ⟂ Z | kept_do_vars, obs_vars in the modified graph
            conditioning_set = kept_do_vars + obs_vars

            try:
                if self._custom_d_separation(Y, Z, conditioning_set, g_test):
                    new_conditions = [c for k, c in enumerate(conditions) if k != z_idx]
                    cand = CausalProbability(Y, *new_conditions)
                    s = str(cand)
                    if s not in seen:
                        seen.add(s)
                        outs.append(cand)
            except Exception:
                pass

        return outs
                    
    def suggest_fix(self):
        """
        Suggests symbolic fixes to the expression based on graph structure
        and d-separation logic, including when no conditioning is needed (P(Y)).
        Avoids suggesting do(Z) if Z is a mediator.
        """
        from probability import Do, CausalProbability

        if not hasattr(self.expression, 'args'):
            return []

        Y = self.expression.args[0]
        conditions = list(self.expression.args[1:])

        do_vars = []
        obs_vars = []

        for cond in conditions:
            if isinstance(cond, Do):
                do_vars.append(str(cond.args[0]))
            else:
                obs_vars.append(str(cond))

        suggestions = []

        # check if all observed variables are d-separated from Y in the do(X) graph
        all_dsep = True
        g_do = self._create_intervention_graph(do_vars)

        for z in obs_vars:
            others = [v for v in obs_vars if v != z]
            if not self._custom_d_separation(str(Y), z, do_vars + others, g_do):
                all_dsep = False
                break

        # If all observed variables are d-separated from Y, suggest dropping them
        if all_dsep and obs_vars:
            suggestions.append(
                f"All observed variables are d-separated from {Y} in the interventional graph. "
                f"Consider using P({Y}) instead — no conditioning is necessary."
            )
            return suggestions

        for z in obs_vars:
            if self.graph.has_edge(z, str(Y)):
                is_mediator = any(
                    nx.has_path(self.graph, x, z)
                    for x in obs_vars if x != z
                )
                if is_mediator:
                    suggestions.append(
                        f"{z} is a mediator between a cause and {Y}. Avoid conditioning on {z} to prevent post-treatment bias."
                    )
                else:
                    suggestions.append(
                        f"{z} causes {Y}, but is only observed. Consider using do({z}) if you intend an intervention."
                    )

            # Check d-separation fails
            others = [v for v in obs_vars if v != z]
            if not self._custom_d_separation(str(Y), z, do_vars + others, g_do):
                suggestions.append(
                    f"Conditioning on {z} may bias results; {Y} is not d-separated from {z} given {do_vars + others}."
                )
            if all_dsep and obs_vars:
                suggestions.append(
                    f"All observed variables are d-separated from {Y} in the interventional graph. "
                    f"Consider using P({Y}) instead — no conditioning is necessary."
    )
        return suggestions



