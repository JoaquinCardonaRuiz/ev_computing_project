import json
from scipy.optimize import dual_annealing
import numpy as np


# Define your objective function
def objective_function(x, config_keys, config):
    # Update the config object with the parameters from x
    for i, key in enumerate(config_keys):
        if key.endswith("_bounds"):
            continue  # Skip the bounds entries
        setattr(config, key, x[i])

    # Create a DEAP_Optimiser object and run the optimise method
    optimiser = DEAP_Optimiser(config)

    # Initialize best fitness at generation 10 to negative infinity
    best_gen_10 = -np.inf

    for gen in range(config.n_generations):
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
with open("config.json", "r") as f:
    config_dict = json.load(f)

# Create a DEAPConfig object from the dictionary
config = DEAPConfig(config_dict)

# Precompile the list of keys and bounds
config_keys = [key for key in config.__dict__.keys() if not key.endswith("_bounds")]
bounds = [getattr(config, key + "_bounds") for key in config_keys]


# Define a local function for the objective
def local_objective(x):
    return objective_function(x, config_keys, config)


# Call dual_annealing()
result = dual_annealing(local_objective, bounds)

# The optimized parameters are in result.x
optimized_params = {key: value for key, value in zip(config_keys, result.x)}
print(optimized_params)
