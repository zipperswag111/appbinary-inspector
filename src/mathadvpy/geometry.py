"""Plane geometry helpers."""
from __future__ import annotations

import math


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def midpoint(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float]:
    return (x1 + x2) / 2, (y1 + y2) / 2


def triangle_area(base: float, height: float) -> float:
    return base * height / 2


def circle_area(radius: float) -> float:
    if radius < 0:
        raise ValueError("radius cannot be negative")
    return math.pi * radius ** 2
