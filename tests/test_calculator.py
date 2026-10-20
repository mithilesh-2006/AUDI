"""
Unit tests for Safe Deterministic Calculator.
Verifies arithmetic correctness, boundary conditions, and strict security sandboxing.
"""

import unittest
from src.calculator import SafeCalculator, CalculationResult


class TestSafeCalculator(unittest.TestCase):

    def test_basic_arithmetic(self):
        res = SafeCalculator.evaluate("14 * 25")
        self.assertTrue(res.success)
        self.assertEqual(res.result, 350)

        res2 = SafeCalculator.evaluate("(120 + 45) / 5")
        self.assertTrue(res2.success)
        self.assertEqual(res2.result, 33)

        res3 = SafeCalculator.evaluate("2 ** 8")
        self.assertTrue(res3.success)
        self.assertEqual(res3.result, 256)

        res4 = SafeCalculator.evaluate("2 ^ 8")  # Caret alias
        self.assertTrue(res4.success)
        self.assertEqual(res4.result, 256)

        res5 = SafeCalculator.evaluate("100 - 45.5")
        self.assertTrue(res5.success)
        self.assertAlmostEqual(res5.result, 54.5)

    def test_math_functions_and_constants(self):
        res = SafeCalculator.evaluate("sqrt(144) + 10")
        self.assertTrue(res.success)
        self.assertEqual(res.result, 22.0)

        res2 = SafeCalculator.evaluate("round(pi, 2)")
        self.assertTrue(res2.success)
        self.assertEqual(res2.result, 3.14)

        res3 = SafeCalculator.evaluate("abs(-50)")
        self.assertTrue(res3.success)
        self.assertEqual(res3.result, 50)

    def test_division_by_zero(self):
        res = SafeCalculator.evaluate("100 / 0")
        self.assertFalse(res.success)
        self.assertIn("Division by zero", res.error)

    def test_empty_and_malformed_input(self):
        res = SafeCalculator.evaluate("")
        self.assertFalse(res.success)
        self.assertIn("Empty", res.error)

        res2 = SafeCalculator.evaluate("   ")
        self.assertFalse(res2.success)

        res3 = SafeCalculator.evaluate("2 + + +")
        self.assertFalse(res3.success)

    def test_dos_large_exponent_rejection(self):
        # Exponents > 1000 should be rejected to prevent Denial of Service
        res = SafeCalculator.evaluate("2 ** 99999")
        self.assertFalse(res.success)
        self.assertIn("exceeds safe threshold", res.error)

    def test_security_rejections(self):
        # Arbitrary code execution attempts must fail
        malicious_inputs = [
            "__import__('os').system('ls')",
            "import os",
            "eval('2 + 2')",
            "open('passwords.txt')",
            "x = 5",
            "[x for x in range(10)]",
            "lambda x: x + 1",
            "__builtins__",
            "os.remove('file')",
            "sys.exit()",
            "'hello'.upper()",
            "1 + True",
        ]
        for expr in malicious_inputs:
            with self.subTest(expr=expr):
                res = SafeCalculator.evaluate(expr)
                self.assertFalse(res.success, f"Expected '{expr}' to be rejected")
                self.assertTrue(
                    res.error is not None and any(w in res.error.lower() for w in ["security", "invalid", "forbidden", "unsupported"]),
                    f"Unexpected error message for '{expr}': {res.error}"
                )


if __name__ == "__main__":
    unittest.main()
