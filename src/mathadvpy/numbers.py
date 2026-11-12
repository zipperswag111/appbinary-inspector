"""Number theory and arithmetic helpers."""
from __future__ import annotations

import math


def gcd(a: int, b: int) -> int:
    return math.gcd(a, b)


def lcm(a: int, b: int) -> int:
    return math.lcm(a, b)


def factorial(value: int) -> int:
    if value < 0 or int(value) != value:
        raise ValueError("factorial needs a non-negative integer")
    return math.factorial(value)


def prime_factors(value: int) -> list[int]:
    """Return prime factors, including repeated factors."""
    value = abs(int(value))
    if value < 2:
        return []
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            factors.append(divisor)
            value //= divisor
        divisor += 1 if divisor == 2 else 2
    if value > 1:
        factors.append(value)
    return factors
