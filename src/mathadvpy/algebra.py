"""Algebra helpers."""
from __future__ import annotations

import cmath


def solve_linear(a: float, b: float) -> float:
    """Solve ax + b = 0."""
    if a == 0:
        raise ValueError("a cannot be zero for a linear equation")
    return -b / a


def solve_quadratic(a: float, b: float, c: float) -> tuple[complex, complex]:
    """Solve ax² + bx + c = 0, returning real or complex roots."""
    if a == 0:
        root = solve_linear(b, c)
        return root, root
    discriminant = b * b - 4 * a * c
    root = cmath.sqrt(discriminant)
    return (-b + root) / (2 * a), (-b - root) / (2 * a)


def evaluate_polynomial(coefficients: list[float], x: float) -> float:
    """Evaluate coefficients ordered highest degree to constant using Horner's method."""
    result = 0.0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result
