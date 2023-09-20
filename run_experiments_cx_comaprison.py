""" Module to run EvoMan experiments.

Write below the logic for the experiments to run, importing optimizers and
loading or generating configurations as needed.
"""
from experiment_utils import (
    process_prefixed_dirs,
    save_sorted_results,
    print_top_n_results_from_file,
)
import json
import itertools

from tqdm import tqdm
import numpy as np

# DEAP optimiser
from deap_algorithm import DEAP_Optimiser


# Test DEAP
config = ""
with open("deap_config_compare_cx.json") as json_file:
    config = json.loads(json_file.read())

exp_template = config["experiments"][0]

# Define the methods and their possible kwargs values
methods = [
    # Working
    {"name": "cxOnePoint", "kwargs": {}},
    # {"name": "cxTwoPoint", "kwargs": {}},
    # {"name": "cxUniform", "kwargs": {"indpb": list(np.arange(0.1, 1.0, 0.1))}},
    # {"name": "cxBlend", "kwargs": {"alpha": list(np.arange(0.1, 1.0, 0.1))}},
    # {"name": "cxSimulatedBinary", "kwargs": {"eta": list(range(1, 15, 1))}},
    # maybe works, dunno low and up
    # {"name": "cxSimulatedBinaryBounded", "kwargs":{"eta": 15.0, "low": [], "up": []}},
    # Not working
    # {"name": "cxPartialyMatched", "kwargs": {}},
    # {"name": "cxUniformPartialyMatched", "kwargs": {"indpb": 0.5}},
    # {"name": "cxOrdered", "kwargs": {}},
    # {"name": "cxMessyOnePoint", "kwargs": {}},
    # {"name": "cxESBlend", "kwargs": {"alpha": 0.5}},
    # {"name": "cxESTwoPoint", "kwargs": {}},
]

# Calculate the total number of combinations for all methods
total_combinations = sum(
    len(list(itertools.product(*method["kwargs"].values()))) for method in methods
)

# Create a progress bar
pbar = tqdm(total=total_combinations, desc="Total progress")


# Set folder prefix string
prefix = "crossover_exp_"

# Iterate over each method
for method in methods:
    # Get the kwargs and their possible values
    kwargs = method["kwargs"]

    # Generate all combinations of kwargs values
    combinations = [
        dict(zip(kwargs.keys(), values))
        for values in itertools.product(*kwargs.values())
    ]

    # Iterate over each combination
    for combination in combinations:
        # Create a copy of the experiment template
        exp = exp_template.copy()

        # Update the experiment parameters
        arg_names = (
            "_".join([f"{k}_{v}" for k, v in combination.items()])
            if combination
            else ""
        )
        exp["experiment_name"] = f"{prefix}{method['name']}_{arg_names}"
        exp["cx_method"] = method["name"]
        exp["cx_kwargs"] = combination

        # Run the experiment
        print(json.dumps(exp, indent=4))
        optimiser = DEAP_Optimiser(exp)
        optimiser.optimise()

        # Update the progress bar
        pbar.update()

start_dir = "."
result_filename = "results_crossover_methods_exp.json"
results = process_prefixed_dirs(start_dir, prefix)
save_sorted_results(results, result_filename)
print_top_n_results_from_file(result_filename, 10)

# Close the progress bar
pbar.close()
