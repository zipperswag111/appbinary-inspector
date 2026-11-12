"""MathAdvPy: practical algebra, geometry, trigonometry, precalculus and calculus."""

from .algebra import evaluate_polynomial, solve_linear, solve_quadratic
from .calculus import derivative, integrate, limit
from .geometry import circle_area, distance, midpoint, triangle_area
from .numbers import factorial, gcd, lcm, prime_factors
from .precalculus import arithmetic_sequence, geometric_sequence
from .statistics import mean, median, standard_deviation
from .trigonometry import cos, degrees, radians, sin, tan

__version__ = "1.0.0"

__all__ = [
    "arithmetic_sequence", "circle_area", "cos", "degrees", "derivative",
    "distance", "evaluate_polynomial", "factorial", "gcd", "geometric_sequence",
    "integrate", "lcm", "limit", "mean", "median", "midpoint", "prime_factors",
    "radians", "sin", "solve_linear", "solve_quadratic", "standard_deviation",
    "tan", "triangle_area",
]
