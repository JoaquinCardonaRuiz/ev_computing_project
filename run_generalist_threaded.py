import json
import threading

# DEAP optimiser
from deap_algorithm import DEAP_Optimiser

def run_experiment(exp):
    name = exp['experiment_name']
    enemy_sets = {'set_1': [1,2]}
    for enemy_set in enemy_sets:
        for i in range(1):
            exp['enemies'] = enemy_sets[enemy_set]
            exp['experiment_name'] = name + '_e_'+enemy_set+'_r'+str(i)
            optimiser = DEAP_Optimiser(exp)
            optimiser.optimise()

# Test DEAP
config = ""
with open('deap_config.json') as json_file:
    config = json.loads(json_file.read())

threads = []
for exp in config['experiments']:
    t = threading.Thread(target=run_experiment, args=(exp,))
    threads.append(t)
    t.start()

# Wait for all threads to finish
for t in threads:
    t.join()
