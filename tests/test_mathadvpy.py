from mathadvpy import arithmetic_sequence, circle_area, derivative, distance, integrate, mean, solve_quadratic, standard_deviation


def test_core_math_topics():
    assert solve_quadratic(1, -5, 6) == (3 + 0j, 2 + 0j)
    assert distance(0, 0, 3, 4) == 5
    assert arithmetic_sequence(2, 3, 4) == [2, 5, 8, 11]
    assert round(circle_area(1), 6) == 3.141593
    assert round(derivative(lambda x: x**2, 3), 5) == 6
    assert round(integrate(lambda x: x**2, 0, 3), 4) == 9
    assert mean([2, 4, 6]) == 4
    assert round(standard_deviation([2, 4, 6]), 5) == 1.63299
