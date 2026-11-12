"""Trigonometry helpers using degrees by default."""
from __future__ import annotations

import math


def radians(angle: float) -> float:
    return math.radians(angle)


def degrees(angle: float) -> float:
    return math.degrees(angle)


def sin(angle: float, *, in_degrees: bool = True) -> float:
    return math.sin(radians(angle) if in_degrees else angle)


def cos(angle: float, *, in_degrees: bool = True) -> float:
    return math.cos(radians(angle) if in_degrees else angle)


def tan(angle: float, *, in_degrees: bool = True) -> float:
    return math.tan(radians(angle) if in_degrees else angle)
