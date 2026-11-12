"""Precalculus sequence helpers."""
from __future__ import annotations


def arithmetic_sequence(first: float, difference: float, terms: int) -> list[float]:
    if terms < 0:
        raise ValueError("terms cannot be negative")
    return [first + index * difference for index in range(terms)]


def geometric_sequence(first: float, ratio: float, terms: int) -> list[float]:
    if terms < 0:
        raise ValueError("terms cannot be negative")
    return [first * ratio ** index for index in range(terms)]
