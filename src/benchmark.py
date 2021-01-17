import numpy as np


def ackley(x: np.ndarray) -> float:
    # https://www.sfu.ca/~ssurjano/ackley.html
    assert x.ndim == 1
    return 20 - 20*np.exp(-0.2*np.sqrt(1/x.shape[0] * np.sum(x**2))) + np.e - np.exp(1/x.shape[0] * np.sum(np.cos(2*np.pi*x)))


def griewank(x: np.ndarray) -> float:
    # https://www.sfu.ca/~ssurjano/griewank.html
    assert x.ndim == 1
    return np.sum(x**2 / 4000) - np.prod(np.cos(x/np.sqrt(np.arange(1, x.shape[0] + 1)))) + 1


def rastrigin(x: np.ndarray) -> float:
    # https://www.sfu.ca/~ssurjano/rastr.html
    assert x.ndim == 1
    return 10*x.shape[0] + np.sum(x**2 - 10*np.cos(2*np.pi*x))
