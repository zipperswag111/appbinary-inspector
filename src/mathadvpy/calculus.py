"""Numerical single-variable calculus helpers."""
from __future__ import annotations

from collections.abc import Callable

Function = Callable[[float], float]


def derivative(function: Function, x: float, *, h: float = 1e-5) -> float:
    """Estimate f′(x) with a central difference."""
    if h <= 0:
        raise ValueError("h must be positive")
    return (function(x + h) - function(x - h)) / (2 * h)


def limit(function: Function, x: float, *, h: float = 1e-6) -> float:
    """Estimate a two-sided limit near x."""
    if h <= 0:
        raise ValueError("h must be positive")
    return (function(x - h) + function(x + h)) / 2


def integrate(function: Function, start: float, end: float, *, intervals: int = 10_000) -> float:
    """Estimate a definite integral using the trapezoid rule."""
    if intervals < 1:
        raise ValueError("intervals must be at least one")
    width = (end - start) / intervals
    total = (function(start) + function(end)) / 2
    for index in range(1, intervals):
        total += function(start + index * width)
    return total * width
