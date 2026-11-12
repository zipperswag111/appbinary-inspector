"""Command line entry point for MathAdvPy."""
from __future__ import annotations

import argparse
import json

from . import circle_area, distance, evaluate_polynomial, solve_linear, solve_quadratic


def main() -> None:
    parser = argparse.ArgumentParser(description="MathAdvPy calculator tools")
    commands = parser.add_subparsers(dest="command", required=True)
    linear = commands.add_parser("linear", help="solve ax + b = 0")
    linear.add_argument("a", type=float)
    linear.add_argument("b", type=float)
    quadratic = commands.add_parser("quadratic", help="solve ax² + bx + c = 0")
    quadratic.add_argument("a", type=float)
    quadratic.add_argument("b", type=float)
    quadratic.add_argument("c", type=float)
    polynomial = commands.add_parser("polynomial", help="evaluate a polynomial")
    polynomial.add_argument("x", type=float)
    polynomial.add_argument("coefficients", type=float, nargs="+")
    geometry = commands.add_parser("distance", help="distance between two points")
    geometry.add_argument("x1", type=float)
    geometry.add_argument("y1", type=float)
    geometry.add_argument("x2", type=float)
    geometry.add_argument("y2", type=float)
    circle = commands.add_parser("circle-area", help="area of a circle")
    circle.add_argument("radius", type=float)
    args = parser.parse_args()
    if args.command == "linear":
        result = solve_linear(args.a, args.b)
    elif args.command == "quadratic":
        roots = solve_quadratic(args.a, args.b, args.c)
        result = {"roots": [str(root) for root in roots]}
    elif args.command == "polynomial":
        result = evaluate_polynomial(args.coefficients, args.x)
    elif args.command == "distance":
        result = distance(args.x1, args.y1, args.x2, args.y2)
    else:
        result = circle_area(args.radius)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
