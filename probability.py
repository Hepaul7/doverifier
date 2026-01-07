import sympy as sp
from sympy import Symbol, Eq, Add, Mul, sympify
import re
from sympy.printing.str import StrPrinter

import unittest


class Do(sp.Function):
    def __new__(cls, var, value=None):
        if value is not None:
            return super().__new__(cls, var, value)
        return super().__new__(cls, var)
    
    def __str__(self):
        if len(self.args) == 1:
            return f'do({self.args[0]})'
        return f'do({self.args[0]}={self.args[1]})'
    
    def __repr__(self):
        return self.__str__()
    
    def _sympystr(self, printer):
        return str(self)

class Mult(sp.Function):
    """
    Represents a product of probability expressions.
    """
    def __new__(cls, *args):
        return sp.Function.__new__(cls, *args)
    
    def __str__(self):
        return ' * '.join(str(arg) for arg in self.args)
    
    def _sympystr(self, printer):
        return self.__str__()

class ProbabilityExpression(sp.Expr):
    """
    Base class for probability expressions that can be part of arithmetic operations.
    """
    pass

def _condition_sort_key(condition):
    """
    Deterministic key for sorting conditions.
    Order groups: Do(...) first, then Eq(...), then everything else.
    """
    if isinstance(condition, Do):
        var = str(condition.args[0])
        val = str(condition.args[1]) if len(condition.args) == 2 else ""
        return (0, var, val)
    if isinstance(condition, Eq):
        return (1, str(condition.lhs), str(condition.rhs))
    return (2, str(condition), "")


class SumOver(sp.Function):
    """
    Represents Σ_{vars} (expr).
    Example: SumOver(Z, expr) or SumOver((Z1,Z2), expr)
    """
    def __new__(cls, vars_, expr):
        # vars_ can be a Symbol or a tuple of Symbols
        if isinstance(vars_, (list, tuple)):
            vars_ = tuple(vars_)
        return sp.Function.__new__(cls, vars_, expr)

    def __str__(self):
        vars_, expr = self.args
        if isinstance(vars_, tuple):
            vstr = ", ".join(str(v) for v in vars_)
        else:
            vstr = str(vars_)
        return f"Σ_{{{vstr}}}[{expr}]"

    def _sympystr(self, printer):
        return self.__str__()
    

class CausalProbability(ProbabilityExpression):
    """
    Represents something similar to P(Y=1 | do(X=0), Z=3).
    Now inherits from ProbabilityExpression to support arithmetic.
    """
    def __new__(cls, outcome, *conditions):
        conditions = tuple(sorted(conditions, key=_condition_sort_key))
        args = (outcome,) + conditions
        obj = sp.Expr.__new__(cls, *args)
        obj._outcome = outcome
        obj._conditions = conditions
        return obj
    
    @property
    def args(self):
        """Return args in the expected format for compatibility."""
        return (self._outcome,) + self._conditions
    
    def __str__(self):
        outcome_str = self._format_outcome(self._outcome)
        if len(self._conditions) == 0:
            return f'P({outcome_str})'
        
        conditions_str = ', '.join(map(self._format_condition, self._conditions))
        return f'P({outcome_str} | {conditions_str})'
    
    def _format_outcome(self, outcome):
        if isinstance(outcome, tuple) and len(outcome) == 2:
            var, val = outcome
            return f"{var}={val}"
        elif isinstance(outcome, Eq):
            return f"{outcome.lhs}={outcome.rhs}"
        return str(outcome)
    
    def _format_condition(self, condition):
        if isinstance(condition, Do):
            return str(condition)
        elif isinstance(condition, tuple) and len(condition) == 2:
            var, val = condition
            return f"{var}={val}"
        elif isinstance(condition, Eq):
            return f"{condition.lhs}={condition.rhs}"
        return str(condition)
    
    def _sympystr(self, printer):
        return str(self)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        """Custom equality to ensure SymPy can distinguish different probability expressions."""
        if not isinstance(other, CausalProbability):
            return False
        return (self._outcome == other._outcome and 
                self._conditions == other._conditions)
    
    def __hash__(self):
        """Custom hash to ensure SymPy can distinguish different probability expressions."""
        return hash((self._outcome, self._conditions))
    
    def _hashable_content(self):
        """SymPy uses this for hashing and comparison."""
        return (self._outcome,) + self._conditions
        
    @classmethod
    def parse(cls, expr_str):
        """
        Enhanced parser that can handle:
        - Basic probability expressions: 'P(Y)', 'P(Y|X)', etc.
        - Arithmetic operations: 'P(Y_{X=1}) - P(Y_{X=0})'
        - Products: 'P(A|B)*P(B)'
        - Complex expressions: 'P(Y_{X=0, V2=1} - Y_{X=0, V2=0})'
        """
        # Remove all spaces for easier parsing
        expr_str = expr_str.replace(' ', '')
        
        # Check if this is an arithmetic expression (contains +, -, *, / outside of P(...))
        if cls._contains_arithmetic_outside_parentheses(expr_str):
            return cls._parse_arithmetic_expression(expr_str)
        
        # Handle simple products (backward compatibility)
        if '*' in expr_str and not cls._is_inside_probability(expr_str, expr_str.find('*')):
            factors = [cls.parse(f.strip()) for f in expr_str.split('*')]
            return Mult(*factors)
        
        # Parse single probability expression
        return cls._parse_single_probability(expr_str)
    
    @classmethod
    def _contains_arithmetic_outside_parentheses(cls, expr_str):
        """Check if the expression contains arithmetic operators outside P(...) expressions."""
        in_prob = False
        paren_depth = 0
        
        for i, char in enumerate(expr_str):
            if char == 'P' and i + 1 < len(expr_str) and expr_str[i + 1] == '(':
                in_prob = True
                paren_depth = 0
            elif in_prob and char == '(':
                paren_depth += 1
            elif in_prob and char == ')':
                paren_depth -= 1
                if paren_depth == 0:
                    in_prob = False
            elif not in_prob and char in '+-*/':
                return True
        
        return False
    
    @classmethod
    def _is_inside_probability(cls, expr_str, pos):
        """Check if a position is inside a P(...) expression."""
        # Count P( before this position
        before = expr_str[:pos]
        p_count = before.count('P(')
        close_count = before.count(')')
        return p_count > close_count
    
    @classmethod
    def _parse_arithmetic_expression(cls, expr_str):
        """Parse arithmetic expressions involving probability terms."""

        # Find all P(...) expressions and replace them with placeholders
        prob_pattern = r'P\([^)]+(?:\|[^)]+)?\)'
        prob_matches = list(re.finditer(prob_pattern, expr_str))
        
        # Create placeholder mapping
        placeholders = {}
        modified_expr = expr_str
        
        for i, match in enumerate(reversed(prob_matches)):  # Reverse to maintain positions
            placeholder = f'PROB_{i}'
            prob_expr = match.group()
            placeholders[placeholder] = cls._parse_single_probability(prob_expr)
            
            # Replace in the expression
            start, end = match.span()
            modified_expr = modified_expr[:start] + placeholder + modified_expr[end:]
        
        # Parse the arithmetic expression using SymPy
        # Replace placeholders with symbols temporarily
        temp_symbols = {}
        temp_expr = modified_expr
        for placeholder in placeholders:
            temp_symbol = Symbol(placeholder)
            temp_symbols[temp_symbol] = placeholders[placeholder]
            temp_expr = temp_expr.replace(placeholder, str(temp_symbol))
        
        try:
            # Parse as SymPy expression
            parsed = sympify(temp_expr)
            
            # Replace symbols back with probability expressions
            result = parsed.subs(temp_symbols)
            return result
            
        except Exception as e:
            raise ValueError(f"Could not parse arithmetic expression: {expr_str}") from e
    
    @classmethod
    def _parse_single_probability(cls, expr_str):
        """Parse a single probability expression (original logic)."""
        # Remove P( and trailing )
        if not expr_str.startswith('P(') or not expr_str.endswith(')'):
            raise ValueError(f"Invalid probability expression format: {expr_str}")
        
        inner = expr_str[2:-1]  # Remove P( and )
        
        # Split by | to separate outcome from conditions
        parts = inner.split('|', 1)
        outcome_part = parts[0]
        
        # Check for subscript notation in outcome
        subscript_pattern = r'(\w+)_{([^}]+)}'
        subscript_match = re.match(subscript_pattern, outcome_part)
        
        subscript_conditions = []
        if subscript_match:
            var_name, subscript_str = subscript_match.groups()
            outcome = cls._parse_variable_assignment(var_name)
            
            # Parse subscript conditions as do() operations
            for cond in subscript_str.split(','):
                cond = cond.strip()
                if '=' in cond:
                    var_str, val_str = cond.split('=', 1)
                    var = Symbol(var_str.strip())
                    val_str = val_str.strip()
                    
                    try:
                        value = float(val_str) if '.' in val_str else int(val_str)
                    except ValueError:
                        value = Symbol(val_str)
                    
                    subscript_conditions.append(Do(var, value))
                else:
                    var = Symbol(cond.strip())
                    subscript_conditions.append(Do(var))
        else:
            outcome = cls._parse_variable_assignment(outcome_part)
        
        # If no conditions after |, return with just subscript conditions
        if len(parts) == 1:
            return cls(outcome, *subscript_conditions)
        
        # Parse conditions after |
        conditions_part = parts[1]
        condition_list = subscript_conditions.copy()
        
        do_pattern = r'do\(([^)]+)\)'
        do_matches = re.finditer(do_pattern, conditions_part)
        
        for match in do_matches:
            do_content = match.group(1) 
            
            variables = [var.strip() for var in do_content.split(',')]
            
            for var_expr in variables:
                if '=' in var_expr:
                    var_name, value_str = var_expr.split('=', 1)
                    var = Symbol(var_name.strip())
                    value_str = value_str.strip()
                    
                    try:
                        value = float(value_str) if '.' in value_str else int(value_str)
                    except ValueError:
                        value = Symbol(value_str)
                    
                    condition_list.append(Do(var, value))
                else:
                    var = Symbol(var_expr.strip())
                    condition_list.append(Do(var))
        
        # Remove do() expressions from conditions_part to handle remaining conditions
        remaining_conditions = re.sub(r'do\([^)]+\)(?:,\s*|$)', '', conditions_part)
        remaining_conditions = re.sub(r'^,\s*|,\s*$', '', remaining_conditions)  # Clean up leading/trailing commas
        
        if remaining_conditions:
            for cond in remaining_conditions.split(','):
                cond = cond.strip()
                if cond:  
                    condition = cls._parse_variable_assignment(cond)
                    condition_list.append(condition)
        
        return cls(outcome, *condition_list)
    
    @staticmethod
    def _parse_variable_assignment(expr):
        """Helper method to parse variable assignments like 'Y=1' or just 'Y'"""
        expr = expr.strip()
        if '=' in expr:
            var_str, val_str = expr.split('=', 1)
            var = Symbol(var_str.strip())
            val_str = val_str.strip()
            
            try:
                if '.' in val_str:
                    val = float(val_str)
                else:
                    val = int(val_str)
            except ValueError:
                val = Symbol(val_str)
                
            return Eq(var, val)
        else:
            return Symbol(expr)

