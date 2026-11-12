"""Basic descriptive statistics."""
from __future__ import annotations

import math


def _values(values) -> list[float]:
    data = [float(value) for value in values]
    if not data:
        raise ValueError("values cannot be empty")
    return data


def mean(values) -> float:
    data = _values(values)
    return sum(data) / len(data)


def median(values) -> float:
    data = sorted(_values(values))
    middle = len(data) // 2
    return data[middle] if len(data) % 2 else (data[middle - 1] + data[middle]) / 2


def standard_deviation(values, *, sample: bool = False) -> float:
    data = _values(values)
    if sample and len(data) < 2:
        raise ValueError("sample standard deviation needs two values")
    average = mean(data)
    divisor = len(data) - 1 if sample else len(data)
    return math.sqrt(sum((value - average) ** 2 for value in data) / divisor)
