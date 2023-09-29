""" Module to run EvoMan experiments.

Write below the logic for the experiments to run, importing optimizers and 
loading or generating configurations as needed.
"""
import json

# DEAP optimiser
from deap_algorithm import DEAP_Optimiser


# Test DEAP
config = ""
with open('deap_config.json') as json_file:
    config = json.loads(json_file.read())


for exp in config['experiments']:
    name = exp['experiment_name']
    for enemy in [2,7,5]:
        for i in range(10):
            exp['enemy'] = enemy
            exp['experiment_name'] = name + '_e'+str(enemy)+'_r'+str(i)
            optimiser = DEAP_Optimiser(exp)
            optimiser.optimise()