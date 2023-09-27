import json
from scipy.optimize import dual_annealing
import numpy as np


# Define your objective function
def objective_function(x, config_keys, config):
    # Update the config dictionary with the parameters from x
    for i, key in enumerate(config_keys):
        config[key] = x[i]

    # Create a DEAP_Optimiser object and run the optimise method
    optimiser = DEAP_Optimiser(config)
    max_fitness = optimiser.optimise()

    # Since we're using dual_annealing which minimizes the objective function,
    # we return the negative of max_fitness to maximize it.
    return -max_fitness


# Read the JSON file
with open("config.json", "r") as f:
    config = json.load(f)

# Precompile the list of keys and bounds
config_keys = [key for key in config.keys() if not key.endswith("_bounds")]
bounds = [config[key + "_bounds"] for key in config_keys]


# Define a local function for the objective
def local_objective(x):
    return objective_function(x, config_keys, config)


# Call dual_annealing()
result = dual_annealing(local_objective, bounds)

# The optimized parameters are in result.x
optimized_params = {key: value for key, value in zip(config_keys, result.x)}
print(optimized_params)
