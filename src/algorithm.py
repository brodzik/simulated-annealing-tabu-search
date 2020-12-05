import numpy as np


def is_neighbour(x1, x2, r):
    """Checks (x1, x2) neighbourhood relation.

    Args:
        x1: first point
        x2: second point
        r: neighbourhood radius

    Returns:
        true, if one point belongs to the neighbourhood of the other.

    """

    assert r > 0

    return np.linalg.norm(x1 - x2) <= r


def random_neighbour(x, r, clip=0):
    """Pseudo-randomly generates a point from the neighbourhood (x, r).

    Args:
        x: neighbourhood center
        r: neighbourhood radius
        clip: limits the coordinates to interval [-clip, clip], 0 = no limit

    Returns:
        newly generated point.

    """

    n = x.shape[0]

    assert n > 0
    assert r > 0
    assert clip >= 0

    # https://mathworld.wolfram.com/HyperspherePointPicking.html
    y = np.random.normal(0, 1, n)
    y = y / np.linalg.norm(y)

    y = np.random.uniform(0, r) * y + x

    if clip > 0:
        y = y.clip(-clip, clip)

    return y


def check_violates_tabu(x, r, tabu):
    """Checks neighbourhood relation for each point in tabu.

    Args:
        x: point to check
        r: neighbourhood radius
        tabu: list of banned points

    Returns:
        true, if there exists any point from tabu that is in the neighbourhood (x, r).

    """

    for x_t in tabu:
        if is_neighbour(x, x_t, r):
            return True

    return False


def random_valid_neighbour(x, r, tabu, clip=0, alpha=1.0):
    """Continuously generates neighbours until at least one doesn't violate tabu condition.

    Args:
        x: neighbourhood center
        r: neighbourhood radius
        tabu: list of banned points
        clip: limits the coordinates to interval [-clip, clip], 0 = no limit
        alpha: tabu neighbourhood radius coefficient 

    Returns:
        newly generated point.

    """

    while True:
        y = random_neighbour(x, r, clip)

        if not check_violates_tabu(y, alpha * r, tabu):
            return y
