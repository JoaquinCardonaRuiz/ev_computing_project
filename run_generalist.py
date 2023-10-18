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
    enemy_sets = {'set_1': [1,2], 'set_2':[2,4,6,7]}
    for enemy_set in enemy_sets:
        for i in range(10):
            exp['enemies'] = enemy_sets[enemy_set]
            exp['experiment_name'] = name + '_e_'+enemy_set+'_r'+str(i)
            optimiser = DEAP_Optimiser(exp)
            optimiser.optimise()