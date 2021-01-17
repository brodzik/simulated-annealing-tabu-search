import numpy as np
from inspect import signature

from numpy import random


class TabuSimulatedAnnealing:
    """ Class representing simulated annealing algorithm with tabu.

                    Args:
                        radius: neighbourhood radius
                        clip: limits the coordinates to interval [-clip, clip], 0 = no limit
                        alpha: tabu neighbourhood radius coefficient

                    Attributes:
                        __radius (float):
                        __alpha (float):
                        __clip (float):
                        __tabu ([]):
                        __tabu_length (int):
                        log ([]):
                        __point (np.ndarray);
                        __score (float):
                        __maximize (bool):
                        __score_function:
                        __max_tabu_tries:

    """

    def __init__(self, radius: float = 1.0, alpha: float = 0.5, clip: float = 0.0, max_tabu_tries: int=1000000):
        assert radius > 0
        assert clip >= 0
        assert alpha >= 0
        assert max_tabu_tries > 0
        self.__radius = radius
        self.__alpha = alpha
        self.__clip = clip
        self.__tabu = []
        self.__tabu_length = 0
        self.log = []
        self.__point = None
        self.__score = None
        self.__score_function = None
        self.__maximize = False
        self.__max_tabu_tries = max_tabu_tries

    def set_maximize(self):
        self.__maximize = True

    def set_minimize(self):
        self.__maximize = False

    def set_start(self, point: np.ndarray, score_function, tabu_length: int = 10) -> None:
        assert len(signature(score_function).parameters) == 1
        assert tabu_length >= 0
        self.__score_function = score_function
        self.__point = point
        self.__score = self.__score_function(self.__point)
        self.__tabu = []
        self.__tabu_length = tabu_length
        self.log = [(point, self.__score)]

    def next(self, temperature: float) -> (np.ndarray, float):
        assert self.__point is not None
        assert self.__score_function is not None

        new_point = self.__random_valid_neighbour()
        score = self.__score_function(new_point)

        self.log.append((new_point, score))
        which_chosen = "none"
        if self.__is_replaced(score, temperature):
            if score > self.__score:
                which_chosen = "better"
            elif score < self.__score:
                which_chosen = "worse"
            self.__point = new_point
            self.__score = score

        self.__tabu.append(self.__point)
        if len(self.__tabu) > self.__tabu_length:
            self.__tabu.pop(0)

        return self.__point, self.__score, which_chosen

    def __is_replaced(self, score: float, temperature: float) -> bool:
        return (self.__maximize and score > self.__score) or (
                not self.__maximize and score < self.__score) or random.uniform(0, 1) < self.__calculate_annealing(
                temperature, score)

    def __calculate_annealing(self, temperature: float, new_score: float) -> float:
        return np.exp(-np.abs(new_score - self.__score) / temperature)

    def __is_neighbour(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        """Checks (x1, x2) neighbourhood relation.

        Args:
            x1: first point
            x2: second point

        Returns:
            true, if one point belongs to the neighbourhood of the other.

        """
        assert x1.ndim == 1
        assert x2.ndim == 1

        # print(np.linalg.norm(x1 - x2))

        return np.linalg.norm(x1 - x2) <= self.__alpha * self.__radius

    def __check_violates_tabu(self, x: np.ndarray) -> bool:
        """Checks neighbourhood relation for each point in tabu.

        Args:
            x: point to check

        Returns:
            true, if there exists any point from tabu that is in the neighbourhood (x, r).

        """

        for x_t in self.__tabu:
            if self.__is_neighbour(x, x_t):
                return True

        return False

    def __random_neighbour(self, x: np.ndarray) -> np.ndarray:
        """Pseudo-randomly generates a point from the neighbourhood (x, r).

        Args:
            x: neighbourhood center

        Returns:
            newly generated point.

        """

        n = x.shape[0]

        assert n > 0

        # https://mathworld.wolfram.com/HyperspherePointPicking.html
        y = np.random.normal(0, 1, n)
        y = y / np.linalg.norm(y)

        y = np.random.uniform(0, self.__radius) * y + x

        if self.__clip > 0:
            y = y.__clip(-self.__clip, self.__clip)

        return y

    def __random_valid_neighbour(self) -> np.ndarray:
        """Continuously generates neighbours until at least one doesn't violate tabu condition.

        Returns:
            newly generated point.

        """

        while True:
            i = 0
            while i < self.__max_tabu_tries:
                y = self.__random_neighbour(self.__point)

                if not self.__check_violates_tabu(y):
                    assert y.ndim == self.__point.ndim

                    return y

                i = i + 1

            if len(self.__tabu) > 0:
                self.__tabu.pop(0)
