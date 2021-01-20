import numpy as np


def constant(A: float) -> float:
    assert A > 0

    return A


def logarithmic(t: int, A: float):
    assert t >= 0
    assert A > 0

    return A / np.log(2 + t)


def geometric(t: int, A: float, B: float) -> float:
    assert t >= 0
    assert A > 0
    assert 0 < B < 1

    return A * B**t


def adaptive(temperature: float, A: float, B: float, n_better: int, n_worse: int):
    temperature = max(temperature, 1e-3)
    assert A > 0
    assert 0 < B < 1
    assert n_better >= 0
    assert n_worse >= 0

    if n_worse == 0 or n_better / n_worse > A:
        return temperature * (1 + B)
    elif n_better == 0 or n_better / n_worse < A:
        return temperature * (1 - B)
    else:
        return temperature
