import json
import os
import random

import numpy as np
from tqdm import tqdm

from algorithm import *
from benchmark import *
from temperature import *


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def save_data(name: str, data: {}):
    with open(name, "w+") as f:
        f.seek(0)
        json.dump(data, f)
        f.truncate()


TEST_FUNCTIONS = {
    "ackley": (ackley, 32.768, 1.0, 0),
    "griewank": (griewank, 600, 10.0, 0),
    "rastrigin": (rastrigin, 5.12, 0.1, 0),
}

TEST_DIMENSIONS = [2, 5, 10]

COOLING_SCHEDULES = {
    "constant": (constant, [10**i for i in range(-1, 5)]),
    "logarithmic": (logarithmic, [10**i for i in range(-1, 5)]),
    "geometric": (geometric, [(10**i, j/100) for i in range(-1, 5) for j in range(90, 100)]),
    "adaptive": (adaptive, [(10**i, j / 2, k / 100) for i in range(-1, 5) for j in range(1, 5) for k in range(1, 11)]),
}

TABU_LENGTHS = [0, 5, 10]

N_SAMPLES = 20

MAX_ITERATIONS = 1e4


def main():
    pbar = tqdm(total=8424)

    data = {}
    for test_name, (test_func, test_clip, test_radius, test_target) in TEST_FUNCTIONS.items():
        data[str(test_name)] = {}
        for test_dim in TEST_DIMENSIONS:
            data[str(test_name)][str(test_dim)] = {}
            for tabu_len in TABU_LENGTHS:
                data[str(test_name)][str(test_dim)][str(tabu_len)] = {}
                for sched_name, (sched_func, sched_params) in COOLING_SCHEDULES.items():
                    data[str(test_name)][str(test_dim)][str(tabu_len)][str(sched_name)] = {}
                    for sched_param_set in sched_params:
                        #print(test_name, test_dim, tabu_len, sched_name, sched_param_set)

                        log = []

                        for seed in range(N_SAMPLES):
                            seed_everything(seed)
                            t = run_test(test_func, test_clip, test_radius, test_target, test_dim, tabu_len, sched_name, sched_param_set)
                            log.append(t)

                        #print(np.min(log), np.max(log), np.mean(log), np.std(log))

                        data[str(test_name)][str(test_dim)][str(tabu_len)][str(sched_name)][str(sched_param_set)] = {}
                        data[str(test_name)][str(test_dim)][str(tabu_len)][str(sched_name)][str(sched_param_set)]["min"] = float(np.min(log))
                        data[str(test_name)][str(test_dim)][str(tabu_len)][str(sched_name)][str(sched_param_set)]["max"] = float(np.max(log))
                        data[str(test_name)][str(test_dim)][str(tabu_len)][str(sched_name)][str(sched_param_set)]["mean"] = float(np.mean(log))
                        data[str(test_name)][str(test_dim)][str(tabu_len)][str(sched_name)][str(sched_param_set)]["std"] = float(np.std(log))

                        save_data("results.json", data)

                        pbar.update()

    pbar.close()


def run_test(test_func, test_clip, test_radius, test_target, test_dim, tabu_len, sched_name, sched_params):
    tsa = TabuSimulatedAnnealing(radius=test_radius, clip=test_clip)
    tsa.set_minimize()
    tsa.set_start(point=np.full(test_dim, test_clip / 2), score_function=test_func, tabu_length=tabu_len)

    t = 0
    n_better = 0
    n_worse = 0

    if sched_name == "adaptive":
        temperature = sched_params[0]

    while t < MAX_ITERATIONS:
        if sched_name == "constant":
            temperature = constant(sched_params)
        elif sched_name == "logarithmic":
            temperature = logarithmic(t, sched_params)
        elif sched_name == "geometric":
            temperature = geometric(t, sched_params[0], sched_params[1])
        elif sched_name == "adaptive":
            temperature = adaptive(temperature, sched_params[1], sched_params[2], n_better, n_worse)

        point, score, diff = tsa.next(temperature)

        if np.linalg.norm(point - test_target) <= test_radius:
            return t

        if diff == "better":
            n_better += 1
        elif diff == "worse":
            n_worse += 1

        t += 1

    return t


if __name__ == "__main__":
    main()
