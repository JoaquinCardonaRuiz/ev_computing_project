""" Module to run EvoMan experiments.

Write below the logic for the experiments to run, importing optimizers and
loading or generating configurations as needed.
"""
import json
import itertools

from tqdm import tqdm

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
    {"name": "cxTwoPoint", "kwargs": {}},
    {"name": "cxUniform", "kwargs": {"indpb": list(np.arange(0.1, 1.0, 0.1))}},
    {"name": "cxBlend", "kwargs": {"alpha": list(np.arange(0.1, 1.0, 0.1))}},
    {"name": "cxSimulatedBinary", "kwargs": {"eta": list(range(1, 15, 1))}},
    # Not working
    # {"name": "cxPartialyMatched", "kwargs": {}},
    # {"name": "cxUniformPartialyMatched", "kwargs": {"indpb": 0.5}},
    # {"name": "cxOrdered", "kwargs": {}},
    # {"name": "cxSimulatedBinaryBounded", "kwargs": {"eta": 15.0, "low": [], "up": []}}, # maybe works, dunno low and up
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
        exp["experiment_name"] = f"crossover_exp_{method['name']}_{arg_names}"
        exp["cx_method"] = method["name"]
        exp["cx_kwargs"] = combination

        # Run the experiment
        print(json.dumps(exp, indent=4))
        optimiser = DEAP_Optimiser(exp)
        optimiser.optimise()

        # Update the progress bar
        pbar.update()

# Close the progress bar
pbar.close()
