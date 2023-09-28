import json
from scipy.optimize import dual_annealing
import numpy as np
import math
from copy import deepcopy

from deap_algorithm import DEAP_Optimiser


def read_config(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_config_keys(config):
    return [
        key
        for key in config.keys()
        if not key.endswith("_bounds") and key + "_bounds" in config
    ]


def get_bounds(config, config_keys):
    return [config[key + "_bounds"] for key in config_keys]


def update_config(x, config_keys, config):
    for i, key in enumerate(config_keys):
        config[key] = x[i]
    config['h_neurons'] = round(config['h_neurons'])
    config['population_size'] = round(config['population_size'])
    config['parents'] = math.ceil(config['population_size'] / 4)*2
    config['children_per_parent'] = round(config['children_per_parent'])
    config['mut_kwargs']['sigma'] = config['sigma']
    config['mut_kwargs']['indpb'] = config['indpb']
    config['parent_sel_kwargs']['tournsize'] = round(config['parent_tournsize'])
    config['survivor_sel_kwargs']['tournsize'] = round(config['survivor_tournsize'])
    return config


def objective_function(x, config_keys, config):
    updated_config = update_config(x, config_keys, config)
    optimiser = DEAP_Optimiser(updated_config)
    best_gen_10 = -np.inf

    for gen in range(int(updated_config["n_generations"])):
        fitness = optimiser.optimise()
        if gen == 10 and fitness < best_gen_10:
            return -fitness
        best_gen_10 = fitness

    return -fitness


def optimize(config_path):
    config = read_config(config_path)
    config_copy = deepcopy(config)
    config_keys = get_config_keys(config)
    bounds = get_bounds(config, config_keys)

    result = dual_annealing(
        lambda x: objective_function(x, config_keys, config_copy), bounds, maxiter=10
    )

    optimized_params = {key: value for key, value in zip(config_keys, result.x)}

    print(optimized_params)


optimize("param_search_config.json")
