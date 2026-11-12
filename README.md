# MathAdvPy

MathAdvPy is a Python math toolkit for algebra, Algebra 2, geometry,
trigonometry, precalculus, calculus, number theory, and statistics.

## Install

```bash
pip install MathAdvPy
```

## Python use

```python
from mathadvpy import solve_quadratic, sin, derivative, integrate

print(solve_quadratic(1, -5, 6))
print(sin(30))
print(derivative(lambda x: x**2, 3))
print(integrate(lambda x: x**2, 0, 3))
```

## Included topics

- Algebra: linear and quadratic equations, polynomial evaluation
- Geometry: distance, midpoint, triangle and circle area
- Trigonometry: degrees, radians, sine, cosine, tangent
- Precalculus: arithmetic and geometric sequences
- Calculus: numerical derivatives, limits, and definite integrals
- Number theory: factors, greatest common divisor, least common multiple
- Statistics: mean, median, population or sample standard deviation

## Command line

```bash
mathadv linear 2 -8
mathadv quadratic 1 -5 6
mathadv polynomial 2 3 0 -4
mathadv distance 0 0 3 4
mathadv circle-area 5
```
