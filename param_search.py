import json
from scipy.optimize import dual_annealing
import numpy as np
from copy import deepcopy

from deap_algorithm import DEAP_Optimiser


# Define your objective function
def objective_function(x, config_keys, config):
    # Create a copy of the config dictionary
    config_copy = deepcopy(config)

    # Update the config_copy dictionary with the parameters from x
    for i, key in enumerate(config_keys):
        if i <= 12:  # Only change values from the 12th index on
            config_copy[key] = int(x[i])
        else:
            config_copy[key] = x[i]

    # Create a DEAP_Optimiser object and run the optimise method
    optimiser = DEAP_Optimiser(config_copy)

    # Initialize best fitness at generation 10 to negative infinity
    best_gen_10 = -np.inf

    for gen in range(int(config_copy["n_generations"])):
        fitness = optimiser.optimise()

        # Check if we're at generation 10
        if gen == 10:
            # If fitness is worse than best_gen_10, stop early and return fitness
            if fitness < best_gen_10:
                return -fitness

            # Otherwise, update best_gen_10 and continue
            best_gen_10 = fitness

    # Since we're using dual_annealing which minimizes the objective function,
    # we return the negative of max_fitness to maximize it.
    return -fitness


# Read the JSON file
with open("param_search_config.json", "r") as f:
    config = json.load(f)


# Precompile the list of keys and bounds
config_keys = [
    key
    for key in config.keys()
    if not key.endswith("_bounds") and key + "_bounds" in config
]
bounds = [config[key + "_bounds"] for key in config_keys]


# Define a local function for the objective
def local_objective(x):
    return objective_function(x, config_keys, config)


# Call dual_annealing()
result = dual_annealing(local_objective, bounds)

# The optimized parameters are in result.x
optimized_params = {key: value for key, value in zip(config_keys, result.x)}
print(optimized_params)
