import os
import random
import numpy as np
from deap import base, creator, tools

from evoman.environment import Environment
from demo_controller import player_controller

# Output layer neurons
OUT_NEURONS=5
# Hidden Layer neurons
HIDDEN_NEURONS = 10
TOTAL_NEURONS = OUT_NEURONS + HIDDEN_NEURONS

#265
# hn * inputs + hn + on * hn + on
TOTAL_WEIGHTS = 215+50

experiment_name = 'deap_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(HIDDEN_NEURONS),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

env.state_to_log()
toolbox = base.Toolbox()

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("attr_float", np.random.uniform,-1,1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=TOTAL_WEIGHTS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    # [0.15,0.22...]
    f,p,e,t = env.play(pcont=np.array(individual))
    return f

toolbox.register("mate", tools.cxBlend)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

def main():
    pop = toolbox.population(n=250)
    # Crossover probability, Mutation probability, and Number of generations
    CXPB, MUTPB, NGEN = 0.5, 0.5, 25

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = [fit]

    for g in range(NGEN):
        print(f'Mean Fitness for Generation {g}: {np.mean([ind.fitness.values[0] for ind in pop])}.')
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2,0.95)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]

        # The population is entirely replaced by the offspring
        pop[:] = offspring
    return pop

main()