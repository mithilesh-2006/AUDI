"""
Safe Deterministic Calculator.

Evaluates mathematical expressions using a restricted Python AST visitor.
Strictly rejects arbitrary code execution, imports, variable lookups,
attribute access, builtins, loops, and dangerous calls.
"""

from __future__ import annotations
import ast
import math
import operator
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass
class CalculationResult:
    """Outcome of a deterministic calculation request."""
    success: bool
    expression: str
    result: Optional[Union[int, float]] = None
    formatted_result: str = ""
    steps: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "expression": self.expression,
            "result": self.result,
            "formatted_result": self.formatted_result,
            "steps": self.steps,
            "latency_ms": round(self.latency_ms, 3),
            "error": self.error,
        }


class SafeCalculator:
    """AST-based restricted math evaluator."""

    # Allowed binary operators
    BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }

    # Allowed unary operators
    UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Whitelisted mathematical functions
    ALLOWED_FUNCTIONS = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "pow": math.pow,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    # Whitelisted mathematical constants
    ALLOWED_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
    }

    MAX_EXPR_LENGTH = 200
    MAX_EXPONENT = 1000

    @classmethod
    def evaluate(cls, expression_str: str) -> CalculationResult:
        """Safely parse and evaluate an arithmetic expression."""
        start_time = time.perf_counter()
        
        if not expression_str or not expression_str.strip():
            return CalculationResult(
                success=False,
                expression=expression_str,
                error="Empty expression",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        expr_clean = expression_str.strip()
        
        # Replace common alternative symbols
        expr_clean = expr_clean.replace("^", "**").replace("×", "*").replace("÷", "/")

        if len(expr_clean) > cls.MAX_EXPR_LENGTH:
            return CalculationResult(
                success=False,
                expression=expression_str,
                error=f"Expression exceeds maximum allowed length of {cls.MAX_EXPR_LENGTH} characters",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        try:
            parsed = ast.parse(expr_clean, mode="eval")
            evaluator = _ASTMathEvaluator(cls.BIN_OPS, cls.UNARY_OPS, cls.ALLOWED_FUNCTIONS, cls.ALLOWED_CONSTANTS, cls.MAX_EXPONENT)
            val = evaluator.visit(parsed.body)
            latency = (time.perf_counter() - start_time) * 1000
            
            # Format nicely
            if isinstance(val, float):
                # If it's a whole number, display as integer if appropriate
                if val.is_integer() and abs(val) < 1e15:
                    formatted = str(int(val))
                else:
                    formatted = f"{val:,.6g}"
            else:
                formatted = f"{val:,}"

            return CalculationResult(
                success=True,
                expression=expression_str,
                result=val,
                formatted_result=formatted,
                steps=f"Evaluated AST node hierarchy safely: {expression_str} = {formatted}",
                latency_ms=latency,
            )
        except ZeroDivisionError:
            return CalculationResult(
                success=False,
                expression=expression_str,
                error="Mathematical error: Division by zero",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except (ValueError, OverflowError) as e:
            return CalculationResult(
                success=False,
                expression=expression_str,
                error=f"Calculation overflow or domain error: {str(e)}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except SecurityException as se:
            return CalculationResult(
                success=False,
                expression=expression_str,
                error=f"Security violation: {str(se)}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        except Exception as e:
            return CalculationResult(
                success=False,
                expression=expression_str,
                error=f"Invalid mathematical expression: {str(e)}",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )


class SecurityException(Exception):
    """Raised when an expression attempts to execute forbidden AST nodes or identifiers."""
    pass


class _ASTMathEvaluator(ast.NodeVisitor):
    def __init__(self, bin_ops, unary_ops, allowed_funcs, allowed_constants, max_exp):
        self.bin_ops = bin_ops
        self.unary_ops = unary_ops
        self.allowed_funcs = allowed_funcs
        self.allowed_constants = allowed_constants
        self.max_exp = max_exp

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise SecurityException(f"Unsupported constant type: {type(node.value).__name__}")

    def visit_BinOp(self, node: ast.BinOp):
        op_type = type(node.op)
        if op_type not in self.bin_ops:
            raise SecurityException(f"Operator {op_type.__name__} is not allowed")
        
        left = self.visit(node.left)
        right = self.visit(node.right)

        # Protect against massive exponentiation DOS attacks like 9**9**9
        if op_type is ast.Pow:
            if abs(right) > self.max_exp:
                raise ValueError(f"Exponent {right} exceeds safe threshold of {self.max_exp}")
            if abs(left) > 1e10 and right > 50:
                raise ValueError("Potential numerical overflow detected in power operation")

        return self.bin_ops[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in self.unary_ops:
            raise SecurityException(f"Unary operator {op_type.__name__} is not allowed")
        operand = self.visit(node.operand)
        return self.unary_ops[op_type](operand)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SecurityException("Complex or indirect function calls are strictly forbidden")
        func_name = node.func.id
        if func_name not in self.allowed_funcs:
            raise SecurityException(f"Function '{func_name}' is not in the allowed math whitelist")
        
        args = [self.visit(arg) for arg in node.args]
        if node.keywords:
            raise SecurityException("Keyword arguments are not supported in math expressions")
        
        return self.allowed_funcs[func_name](*args)

    def visit_Name(self, node: ast.Name):
        if node.id in self.allowed_constants:
            return self.allowed_constants[node.id]
        raise SecurityException(f"Identifier '{node.id}' is not recognized or permitted")

    def generic_visit(self, node: ast.AST):
        raise SecurityException(f"AST node '{type(node).__name__}' is strictly forbidden in calculator expressions")
