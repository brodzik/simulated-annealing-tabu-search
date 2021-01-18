"""
TODO:
    - dla 2D:
        * uzyc parametrow i dla kazdej funkcji kilkadziesiat razy:
            wybrac punkt
            dla kazdego schematu znalezc srednia liczbe iteracji potrzebnych do dojscia do optimum dla kazdej funkcji
        * stworzyc wykresy porownujace schematy dojscia z jednego wybranego punktu
    - dla ND:
        * uzyc parametrow i dla kazdego schematu znalezc srednia liczbe iteracji potrzebnych do dojscia do optimum dla kazdej funkcji
"""
import json

import matplotlib.pyplot as plt

from src.algorithm import TabuSimulatedAnnealing
from src.benchmark import *
from src.temperature import *

from datetime import datetime

MIN_POINT_COLOR = "#111111"

POINTS_ALPHA = 0.7

COLOR_1 = '#1f77b4'
COLOR_2 = '#bcbd22'
COLOR_3 = '#e377c2'
COLOR_4 = '#9467bd'

TAB_LEN = 10

START_POINT = np.array([10, 10])

BEST_PARAMS_ITERATIONS = 1000

AVG_ITERATIONS = 100000

TAB_LEN_PARAM = [i for i in range(0, 16, 5)]

DIMENSIONS = [2, 5, 10]

GLOBAL_EXTREMUM_RADIUS = 10

POINTS_NUM = 50

PARAM_FILE = "params.json"
AVG_FILE = "avg.json"

MAX_AVG_ITERATIONS = 50

ACCEPTANCE_RADIUS = 0.5


def save_data(name: str, data: {}):
    with open(name, "w+") as f:
        f.seek(0)
        json.dump(data, f)
        f.truncate()


def read_data(name: str):
    with open(name, "r+") as f:
        f.seek(0)
        data = json.load(f)
    return data


def collect_avg_iterations() -> {}:
    all = read_data("all_" + PARAM_FILE)

    # ackley
    ackley_params = all["ackley"]
    for dim in ackley_params:
        optimum = np.array([0 for _ in range(int(dim))])
        for tab_len in ackley_params[dim]:
            points = [np.random.rand(int(dim)) * GLOBAL_EXTREMUM_RADIUS for _ in range(POINTS_NUM)]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("ackley " + str(dim) + " " + str(tab_len))
            # const
            ackley_params[dim][tab_len]["const"] = best_const_avg(points, ackley_params[dim][tab_len]["const"], optimum,
                                                                  ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("ackley_" + AVG_FILE, ackley_params)
            # adaptive
            ackley_params[dim][tab_len]["adaptive"] = best_adaptive_avg(points, ackley_params[dim][tab_len]["adaptive"],
                                                                        optimum, ACCEPTANCE_RADIUS, ackley,
                                                                        int(tab_len))
            save_data("ackley_" + AVG_FILE, ackley_params)
            # logarithmic
            ackley_params[dim][tab_len]["logarithmic"] = best_logarithmic_avg(points, ackley_params[dim][tab_len][
                "logarithmic"], optimum, ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("ackley_" + AVG_FILE, ackley_params)
            # geometrical
            ackley_params[dim][tab_len]["geometrical"] = best_geometrical_avg(points, ackley_params[dim][tab_len][
                "geometrical"], optimum, ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("ackley_" + AVG_FILE, ackley_params)

    # griewank
    griewank_params = all["griewank"]
    for dim in griewank_params:
        optimum = np.array([0 for _ in range(int(dim))])
        for tab_len in griewank_params[dim]:
            points = [np.random.rand(int(dim)) * GLOBAL_EXTREMUM_RADIUS for _ in range(POINTS_NUM)]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("griewank " + str(dim) + " " + str(tab_len))
            # const
            griewank_params[dim][tab_len]["const"] = best_const_avg(points, griewank_params[dim][tab_len]["const"],
                                                                    optimum,
                                                                    ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("griewank_" + AVG_FILE, griewank_params)
            # adaptive
            griewank_params[dim][tab_len]["adaptive"] = best_adaptive_avg(points,
                                                                          griewank_params[dim][tab_len]["adaptive"],
                                                                          optimum, ACCEPTANCE_RADIUS, ackley,
                                                                          int(tab_len))
            save_data("griewank_" + AVG_FILE, griewank_params)
            # logarithmic
            griewank_params[dim][tab_len]["logarithmic"] = best_logarithmic_avg(points, griewank_params[dim][tab_len][
                "logarithmic"], optimum, ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("griewank_" + AVG_FILE, griewank_params)
            # geometrical
            griewank_params[dim][tab_len]["geometrical"] = best_geometrical_avg(points, griewank_params[dim][tab_len][
                "geometrical"], optimum, ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("griewank_" + AVG_FILE, griewank_params)

    # rastrigin
    rastrigin_params = all["rastrigin"]
    for dim in rastrigin_params:
        optimum = np.array([0 for _ in range(int(dim))])
        for tab_len in rastrigin_params[dim]:
            points = [np.random.rand(int(dim)) * GLOBAL_EXTREMUM_RADIUS for _ in range(POINTS_NUM)]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("rastrigin " + str(dim) + " " + str(tab_len))
            # const
            rastrigin_params[dim][tab_len]["const"] = best_const_avg(points, rastrigin_params[dim][tab_len]["const"],
                                                                     optimum,
                                                                     ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("rastrigin_" + AVG_FILE, rastrigin_params)
            # adaptive
            rastrigin_params[dim][tab_len]["adaptive"] = best_adaptive_avg(points,
                                                                           rastrigin_params[dim][tab_len]["adaptive"],
                                                                           optimum, ACCEPTANCE_RADIUS, ackley,
                                                                           int(tab_len))
            save_data("rastrigin_" + AVG_FILE, rastrigin_params)
            # logarithmic
            rastrigin_params[dim][tab_len]["logarithmic"] = best_logarithmic_avg(points, rastrigin_params[dim][tab_len][
                "logarithmic"], optimum, ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("rastrigin_" + AVG_FILE, rastrigin_params)
            # geometrical
            rastrigin_params[dim][tab_len]["geometrical"] = best_geometrical_avg(points, rastrigin_params[dim][tab_len][
                "geometrical"], optimum, ACCEPTANCE_RADIUS, ackley, int(tab_len))
            save_data("rastrigin_" + AVG_FILE, rastrigin_params)

    all["ackley"] = ackley_params
    all["griewank"] = griewank_params
    all["rastrigin"] = rastrigin_params
    save_data("all_" + AVG_FILE, all)

    return all


def best_adaptive_avg(points, params, optimum, radius, function, tab_len, minimize: bool = True):
    avg_iterations = 0

    for i in range(len(points)):
        print("adaptive ", i + 1, " out of ", len(points))
        tsa = TabuSimulatedAnnealing()

        if minimize:
            tsa.set_minimize()
        else:
            tsa.set_maximize()

        first_point = points[i]

        tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)
        t = 1
        temperature = 1
        better = 0
        worse = 0
        lowest_iterations = None
        while t < AVG_ITERATIONS:
            temperature1 = adaptive(temperature, params[0], params[1], better, worse)
            point, score, diff = tsa.next(temperature=temperature1)

            if np.linalg.norm(point - optimum) <= radius:
                if lowest_iterations is None or t < lowest_iterations:
                    lowest_iterations = t
                break

            if diff is "better":
                better = better + 1
            elif diff is "worse":
                worse = worse + 1
            t = t + 1
        if lowest_iterations is None:
            lowest_iterations = t
        avg_iterations = avg_iterations + lowest_iterations

    avg_iterations = avg_iterations / len(points)

    return avg_iterations


def best_logarithmic_avg(points, param, optimum, radius, function, tab_len, minimize: bool = True):
    avg_iterations = 0

    for i in range(len(points)):
        print("logarithmic ", i + 1, " out of ", len(points))
        tsa = TabuSimulatedAnnealing()

        if minimize:
            tsa.set_minimize()
        else:
            tsa.set_maximize()

        first_point = points[i]

        tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)
        t = 1
        lowest_iterations = None
        while t < AVG_ITERATIONS:
            point, score, _ = tsa.next(temperature=logarithmic(t, param))

            if np.linalg.norm(point - optimum) <= radius:
                if lowest_iterations is None or t < lowest_iterations:
                    lowest_iterations = t
                break
            t = t + 1
        if lowest_iterations is None:
            lowest_iterations = t
        avg_iterations = avg_iterations + lowest_iterations

    avg_iterations = avg_iterations / len(points)

    return avg_iterations


def best_geometrical_avg(points, params, optimum, radius, function, tab_len, minimize: bool = True):
    avg_iterations = 0

    for i in range(len(points)):
        print("geometrical ", i + 1, " out of ", len(points))
        tsa = TabuSimulatedAnnealing()

        if minimize:
            tsa.set_minimize()
        else:
            tsa.set_maximize()

        first_point = points[i]

        tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)
        t = 1
        lowest_iterations = None
        while t < AVG_ITERATIONS:
            point, score, _ = tsa.next(temperature=geometrical(t, params[0], params[1]))

            if np.linalg.norm(point - optimum) <= radius:
                if lowest_iterations is None or t < lowest_iterations:
                    lowest_iterations = t
                break
            t = t + 1
        if lowest_iterations is None:
            lowest_iterations = t
        avg_iterations = avg_iterations + lowest_iterations

    avg_iterations = avg_iterations / len(points)

    return avg_iterations


def best_const_avg(points, param, optimum, radius, function, tab_len, minimize: bool = True):
    avg_iterations = 0

    for i in range(len(points)):
        print("const ", i + 1, " out of ", len(points))
        tsa = TabuSimulatedAnnealing()

        if minimize:
            tsa.set_minimize()
        else:
            tsa.set_maximize()

        first_point = points[i]

        tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)
        t = 1
        lowest_iterations = None
        while t < AVG_ITERATIONS:
            point, score, _ = tsa.next(temperature=const(param))

            if np.linalg.norm(point - optimum) <= radius:
                if lowest_iterations is None or t < lowest_iterations:
                    lowest_iterations = t
                break
            t = t + 1
        if lowest_iterations is None:
            lowest_iterations = t
        avg_iterations = avg_iterations + lowest_iterations

    avg_iterations = avg_iterations / len(points)

    return avg_iterations


def collect_params_data() -> {}:
    all = {}
    all["ackley"] = read_data("ackley_" + PARAM_FILE)
    all["griewank"] = read_data("griewank_" + PARAM_FILE)
    all["rastrigin"] = read_data("rastrigin_" + PARAM_FILE)
    save_data("all_" + PARAM_FILE, all)
    return all


def collect_params() -> {}:
    # ackley
    ackley_best_param = {}
    for dim in DIMENSIONS:
        ackley_best_param[dim] = {}
        for tab_len in TAB_LEN_PARAM:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("ackley_" + str(dim) + " " + str(tab_len))
            ackley_best_param[dim][tab_len] = {}
            # const
            ackley_best_param[dim][tab_len]["const"] = best_const_params(ackley, tab_len, dim)
            save_data("ackley_" + PARAM_FILE, ackley_best_param)
            # adaptive
            ackley_best_param[dim][tab_len]["adaptive"] = best_adaptive_params(ackley, tab_len, dim)
            save_data("ackley_" + PARAM_FILE, ackley_best_param)
            # logarithmic
            ackley_best_param[dim][tab_len]["logarithmic"] = best_logarithmic_params(ackley, tab_len, dim)
            save_data("ackley_" + PARAM_FILE, ackley_best_param)
            # geometrical
            ackley_best_param[dim][tab_len]["geometrical"] = best_geometrical_params(ackley, tab_len, dim)
            save_data("ackley_" + PARAM_FILE, ackley_best_param)

    # griewank
    griewank_best_param = {}
    for dim in DIMENSIONS:
        griewank_best_param[dim] = {}
        for tab_len in TAB_LEN_PARAM:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("griewank_" + str(dim) + " " + str(tab_len))
            griewank_best_param[dim][tab_len] = {}
            # const
            griewank_best_param[dim][tab_len]["const"] = best_const_params(griewank, tab_len, dim)
            save_data("griewank_" + PARAM_FILE, griewank_best_param)
            # adaptive
            griewank_best_param[dim][tab_len]["adaptive"] = best_adaptive_params(griewank, tab_len, dim)
            save_data("griewank_" + PARAM_FILE, griewank_best_param)
            # logarithmic
            griewank_best_param[dim][tab_len]["logarithmic"] = best_logarithmic_params(griewank, tab_len, dim)
            save_data("griewank_" + PARAM_FILE, griewank_best_param)
            # geometrical
            griewank_best_param[dim][tab_len]["geometrical"] = best_geometrical_params(griewank, tab_len, dim)
            save_data("griewank_" + PARAM_FILE, griewank_best_param)

    # rastrigin
    rastrigin_best_param = {}
    for dim in DIMENSIONS:
        rastrigin_best_param[dim] = {}
        for tab_len in TAB_LEN_PARAM:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Current Time =", current_time)
            print("rastrigin_" + str(dim) + " " + str(tab_len))
            rastrigin_best_param[dim][tab_len] = {}
            # const
            rastrigin_best_param[dim][tab_len]["const"] = best_const_params(rastrigin, tab_len, dim)
            save_data("rastrigin_" + PARAM_FILE, rastrigin_best_param)
            # adaptive
            rastrigin_best_param[dim][tab_len]["adaptive"] = best_adaptive_params(rastrigin, tab_len, dim)
            save_data("rastrigin_" + PARAM_FILE, rastrigin_best_param)
            # logarithmic
            rastrigin_best_param[dim][tab_len]["logarithmic"] = best_logarithmic_params(rastrigin, tab_len, dim)
            save_data("rastrigin_" + PARAM_FILE, rastrigin_best_param)
            # geometrical
            rastrigin_best_param[dim][tab_len]["geometrical"] = best_geometrical_params(rastrigin, tab_len, dim)
            save_data("rastrigin_" + PARAM_FILE, rastrigin_best_param)

    return collect_params_data()


def best_adaptive_params(function, tab_len, dimension: int = 2, minimize: bool = True):
    best_params = None
    best_sum = None

    points = [np.random.rand(dimension) * GLOBAL_EXTREMUM_RADIUS for _ in range(POINTS_NUM)]

    for params in ADAPTIVE_PARAMS:
        param_sum = 0
        print(params)
        for i in range(POINTS_NUM):
            tsa = TabuSimulatedAnnealing()

            if minimize:
                tsa.set_minimize()
            else:
                tsa.set_maximize()

            first_point = points[i]

            tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)
            t = 1
            temperature = 1
            better = 0
            worse = 0
            while t < BEST_PARAMS_ITERATIONS:
                temperature1 = adaptive(temperature, params[0], params[1], better, worse)
                _, score, diff = tsa.next(temperature=temperature1)
                param_sum += score
                if diff is "better":
                    better = better + 1
                elif diff is "worse":
                    worse = worse + 1
                t = t + 1
        if minimize:
            if best_sum is None or param_sum < best_sum:
                best_sum = param_sum
                best_params = params
        else:
            if best_sum is None or param_sum > best_sum:
                best_sum = param_sum
                best_params = params

    return best_params


def best_logarithmic_params(function, tab_len, dimension: int = 2, minimize: bool = True):
    best_A = None
    best_sum = None

    points = [np.random.rand(dimension) * GLOBAL_EXTREMUM_RADIUS for _ in range(POINTS_NUM)]

    for A in LOGARITHMIC_PARAMS:
        param_sum = 0
        print(A)
        for i in range(POINTS_NUM):
            tsa = TabuSimulatedAnnealing()

            if minimize:
                tsa.set_minimize()
            else:
                tsa.set_maximize()

            first_point = points[i]

            tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)

            t = 1
            while t < BEST_PARAMS_ITERATIONS:
                _, score, _ = tsa.next(temperature=logarithmic(t, A))
                param_sum += score
                t = t + 1
        if minimize:
            if best_sum is None or param_sum < best_sum:
                best_sum = param_sum
                best_A = A
        else:
            if best_sum is None or param_sum > best_sum:
                best_sum = param_sum
                best_A = A

    return best_A


def best_geometrical_params(function, tab_len, dimension: int = 2, minimize: bool = True):
    best_params = None
    best_sum = None

    points = [np.random.rand(dimension) * GLOBAL_EXTREMUM_RADIUS for _ in range(POINTS_NUM)]

    for params in GEOMETRICAL_PARAM:
        param_sum = 0
        print(params)
        for i in range(POINTS_NUM):
            tsa = TabuSimulatedAnnealing()

            if minimize:
                tsa.set_minimize()
            else:
                tsa.set_maximize()

            first_point = points[i]

            tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)

            t = 1
            while t < BEST_PARAMS_ITERATIONS:
                _, score, _ = tsa.next(temperature=geometrical(t, params[0], params[1]))
                param_sum += score
                t = t + 1
        if minimize:
            if best_sum is None or param_sum < best_sum:
                best_sum = param_sum
                best_params = params
        else:
            if best_sum is None or param_sum > best_sum:
                best_sum = param_sum
                best_params = params

    return best_params


def best_const_params(function, tab_len, dimension: int = 2, minimize: bool = True):
    best_A = None
    best_sum = None

    points = [np.random.rand(dimension) * GLOBAL_EXTREMUM_RADIUS for _ in range(POINTS_NUM)]

    for A in CONST_PARAMS:
        param_sum = 0
        print(A)
        for i in range(POINTS_NUM):
            tsa = TabuSimulatedAnnealing()

            if minimize:
                tsa.set_minimize()
            else:
                tsa.set_maximize()

            first_point = points[i]

            tsa.set_start(point=first_point, score_function=function, tabu_length=tab_len)

            t = 1
            while t < BEST_PARAMS_ITERATIONS:
                _, score, _ = tsa.next(temperature=const(A))
                param_sum += score
                t = t + 1
        if minimize:
            if best_sum is None or param_sum < best_sum:
                best_sum = param_sum
                best_A = A
        else:
            if best_sum is None or param_sum > best_sum:
                best_sum = param_sum
                best_A = A

    return best_A


def const_params_diff_2D():
    A1 = 0.1
    A2 = 10.0

    tsa1, tsa2 = prepare_algorithm_diff()

    t = 1
    while t < BEST_PARAMS_ITERATIONS:
        tsa1.next(temperature=const(A1))
        tsa2.next(temperature=const(A2))
        t = t + 1

    draw_diff(tsa1.log, tsa2.log, "A=" + str(A1), "A=" + str(A2), "Schemat chłodzenia: stały", "const")


def geometrical_params_diff_2D():
    A = 1
    A1 = 0.1
    A2 = 10.0

    B = 0.95
    B1 = 0.8
    B2 = 0.999

    tsa1, tsa2 = prepare_algorithm_diff()

    t = 1
    while t < BEST_PARAMS_ITERATIONS:
        tsa1.next(temperature=geometrical(t, A1, B))
        tsa2.next(temperature=geometrical(t, A2, B))
        t = t + 1

    draw_diff(tsa1.log, tsa2.log, "A=" + str(A1), "A=" + str(A2), "Schemat chłodzenia: wykładniczy", "geometrical_A")

    tsa1, tsa2 = prepare_algorithm_diff()

    t = 1
    while t < BEST_PARAMS_ITERATIONS:
        tsa1.next(temperature=geometrical(t, A, B1))
        tsa2.next(temperature=geometrical(t, A, B2))
        t = t + 1

    draw_diff(tsa1.log, tsa2.log, "B=" + str(B1), "B=" + str(B2), "Schemat chłodzenia: wykładniczy", "geometrical_B")


def logarithmic_params_diff_2D():
    A = 1
    A1 = 0.1
    A2 = 10.0

    tsa1, tsa2 = prepare_algorithm_diff()

    t = 1
    while t < BEST_PARAMS_ITERATIONS:
        tsa1.next(temperature=logarithmic(t, A1))
        tsa2.next(temperature=logarithmic(t, A2))
        t = t + 1

    draw_diff(tsa1.log, tsa2.log, "A=" + str(A1), "A=" + str(A2), "Schemat chłodzenia: logarytmiczny", "logarithmic")


def adaptive_params_diff_2D():
    A = 2
    A1 = 0.25
    A2 = 4.0

    B = 0.05
    B1 = 0.001
    B2 = 0.3

    tsa1, tsa2 = prepare_algorithm_diff()

    t = 1
    temperature1 = 1
    temperature2 = 1
    better1 = 0
    better2 = 0
    worse1 = 0
    worse2 = 0
    while t < BEST_PARAMS_ITERATIONS:
        temperature1 = adaptive(temperature1, A1, B, better1, worse1)
        _, _, diff = tsa1.next(temperature=temperature1)
        if diff is "better":
            better1 = better1 + 1
        elif diff is "worse":
            worse1 = worse1 + 1
        temperature2 = adaptive(temperature2, A2, B, better2, worse2)
        _, _, diff = tsa2.next(temperature=temperature2)
        if diff is "better":
            better2 = better2 + 1
        elif diff is "worse":
            worse2 = worse2 + 1
        t = t + 1

    draw_diff(tsa1.log, tsa2.log, "A=" + str(A1), "A=" + str(A2), "Schemat chłodzenia: adaptacyjny", "adaptive_A")

    tsa1, tsa2 = prepare_algorithm_diff()

    t = 1
    temperature1 = 1
    temperature2 = 1
    better1 = 0
    better2 = 0
    worse1 = 0
    worse2 = 0
    while t < BEST_PARAMS_ITERATIONS:
        temperature1 = adaptive(temperature1, A, B1, better1, worse1)
        _, _, diff = tsa1.next(temperature=temperature1)
        if diff is "better":
            better1 = better1 + 1
        elif diff is "worse":
            worse1 = worse1 + 1
        temperature2 = adaptive(temperature2, A, B2, better2, worse2)
        _, _, diff = tsa2.next(temperature=temperature2)
        if diff is "better":
            better2 = better2 + 1
        elif diff is "worse":
            worse2 = worse2 + 1
        t = t + 1

    draw_diff(tsa1.log, tsa2.log, "B=" + str(B1), "B=" + str(B2), "Schemat chłodzenia: adaptacyjny", "adaptive_B")


def prepare_algorithm_diff():
    tsa1 = TabuSimulatedAnnealing()
    tsa2 = TabuSimulatedAnnealing()

    tsa1.set_minimize()
    tsa2.set_minimize()

    first_point = START_POINT

    tsa1.set_start(point=first_point, score_function=ackley, tabu_length=TAB_LEN)
    tsa2.set_start(point=first_point, score_function=ackley, tabu_length=TAB_LEN)

    return tsa1, tsa2


def draw_diff(log1, log2, label1: str, label2: str, title: str = "", name: str = None):
    log1 = np.vstack(log1)
    log2 = np.vstack(log2)

    points, z1 = log1.T
    x, y = np.vstack(points).T
    plt.scatter(x, y, c=COLOR_1, s=2, label=label1, alpha=POINTS_ALPHA)

    points, z2 = log2.T
    x, y = np.vstack(points).T
    plt.scatter(x, y, c=COLOR_2, s=2, label=label2, alpha=POINTS_ALPHA)

    plt.scatter(0, 0, s=20, c=MIN_POINT_COLOR)

    plt.legend()
    plt.grid(True)
    plt.title(title)
    if name is None:
        plt.show()
    else:
        plt.savefig(name + "_traversal.png")

    plt.clf()

    plt.plot(z1, label=label1, c=COLOR_1)
    plt.plot(z2, label=label2, c=COLOR_2)
    plt.legend()
    plt.title(title)
    plt.ylabel("Wartość funkcji celu")
    plt.xlabel("Iteracja")
    if name is None:
        plt.show()
    else:
        plt.savefig(name + "_score.png")


def print_progress_bar(iteration: int, total: int, length: int = 50) -> None:
    filled = int(length * iteration // total)
    bar = '█' * filled + '-' * (length - filled)
    print('\r', f"Progress: |{bar}| " + "{0:.2f}".format(100 * (iteration / float(total))) + "% Iteration: " + str(
        iteration) + "/" + str(total), end='')
    if iteration == total:
        print("\n")


collect_avg_iterations()
