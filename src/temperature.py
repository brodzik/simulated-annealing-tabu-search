import numpy as np

"""
    A - 10^(-3) -> 10^3
"""
CONST_PARAMS = [np.power(10.0, i) for i in range(-3, 4)]


def const(A: float) -> float:
    assert A > 0.0
    return A


"""
    A - 10^(-3) -> 10^3
    B - 0.91 -> 0.99
"""
GEOMETRICAL_PARAM = [(np.power(10.0, A), 0.91 + 0.02 * B) for A in range(-3, 4) for B in range(5)]


def geometrical(t: int, A: float, B: float) -> float:
    assert 1.0 > B > 0
    assert A > 0.0
    assert t >= 0
    return A * np.power(B, t)


"""
    A - 1.0 -> 2.0
"""
LOGARITHMIC_PARAMS = [1.0 + 0.1 * A for A in range(11)]


def logarithmic(t: int, A: float):
    assert A > 0.0
    assert t >= 0
    return A / np.log(1 + t)


"""
    A - 0.5 -> 2.0
    B - 0.01 -> 0.1
"""
ADAPTIVE_PARAMS = [(0.5 + 0.5 * A, 0.01 + 0.01 * B) for A in range(4) for B in range(10)]


def adaptive(temperature: float, A: float, B: float, better: int, worse: int):
    assert 1.0 > B >= 0.0
    assert A > 0.0
    assert better >= 0
    assert worse >= 0
    assert temperature > 0.0
    if worse == 0 or (better / worse) > A:
        return temperature * (1 + B)
    elif better == 0 or (better / worse) < A:
        return temperature * (1 - B)
    else:
        return temperature
