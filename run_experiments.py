""" Module to run EvoMan experiments.

Write below the logic for the experiments to run, importing optimizers and
loading or generating configurations as needed.
"""
import json

# DEAP optimiser
from deap_algorithm import DEAP_Optimiser


# Test DEAP
config = ""
with open("deap_config.json") as json_file:
    config = json.loads(json_file.read())

for exp in config["experiments"]:
    optimiser = DEAP_Optimiser(exp)
    optimiser.optimise()
