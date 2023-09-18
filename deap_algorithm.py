""" EvoMan project - DEAP code

This is the script that runs the EvoMan assignment for Evolutionary Computing using
the DEAP framework. It contains a class named DEAP_Optimiser, which needs to be initialised
with a JSON config string before the 'optimise' method can be called to run the algorithm.

The configuration JSON strings determines whether the algorithm returns or logs its results.
"""
# Built-in imports
import os
import json
import random
import numpy as np
from operator import attrgetter

# DEAP imports
from deap import base, creator, tools
from deap.tools.selection import __all__ as all_selections
from deap.tools.mutation import __all__ as all_mutations
from deap.tools.crossover import __all__ as all_crossovers

# Evoman Imports
from evoman.environment import Environment
from demo_controller import player_controller


def elitistRoulette(individuals, k, elitist_k, fit_attr="fitness"):
    """Select *k* individuals from the input *individuals* using *k*
    spins of a roulette, and deterministically selecting the elitist_k best.
    The selection is made by looking only at the first fitness value of each individual.
    The list returned contains references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param elitist_k: The number of individuals of the top of the fitness list that get
        deterministically selected.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.

    .. warning::
       The roulette selection by definition cannot be used for minimization
       or when the fitness can be smaller or equal to 0.
    """
    s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=True)
    elitist_inds = s_inds[:elitist_k]
    s_inds = s_inds[elitist_k:]
    sum_fits = sum(getattr(ind, fit_attr).values[0] for ind in individuals)
    chosen = elitist_inds
    for i in range(k - elitist_k):
        u = random.random() * sum_fits
        sum_ = 0
        for ind in s_inds:
            sum_ += getattr(ind, fit_attr).values[0]
            if sum_ > u:
                chosen.append(ind)
                break
    return chosen


class DEAP_Optimiser:
    """DEAP Optimiser class

    Runs an evolutionary algorithm using the DEAP framework. Initialises using a
    JSON string that sets the configuration for the class, including hyperparameters
    of the algorithm, as well as methods of selection, mutation, and crossover.

    Contains the following methods:

    :__init__(self, config): initialises the class, takes a JSON string with parameters
        for configuration.
    :set_env(self): configures EvoMan environment.
    :set_toolbox(self): configures DEAP framework with paramenters from the config string.
    :evaluate(self, individual): runs the EvoMan simulation and returns fitness of the individual.
    :optimise(self): main method, runs the whole algorithm for the number of generations set
        in the config, and returns or saves to disk resulting population and statistics.
    """

    def __init__(self, config):
        self.config = config

        # Calculates number of weights
        n_inputs, n_outputs = 20, 5
        self.tot_neurons = (
            self.config["h_neurons"] * (n_inputs + n_outputs + 1) + n_outputs
        )

        # Set simulation Environment
        self.env = self.set_env()

        # Set up DEAP toolbox
        self.toolbox = self.set_toolbox()

    def set_env(self):
        """Initialises EvoMan simulation environment."""
        if not os.path.exists(self.config["experiment_name"]):
            os.makedirs(self.config["experiment_name"])

        env = Environment(
            experiment_name=self.config["experiment_name"],
            enemies=[8],
            playermode="ai",
            player_controller=player_controller(self.config["h_neurons"]),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
        )
        env.state_to_log()
        return env

    def set_toolbox(self):
        """Initialises DEAP framework with specified parameters."""
        toolbox = base.Toolbox()

        cx_methods = {
            method_name: getattr(tools, method_name) for method_name in all_crossovers
        }
        mut_methods = {
            method_name: getattr(tools, method_name) for method_name in all_mutations
        }
        sel_methods = {
            method_name: getattr(tools, method_name) for method_name in all_selections
        }
        sel_methods["elitistRoulette"] = elitistRoulette

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            toolbox.attr_float,
            n=self.tot_neurons,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register(
            "mate", cx_methods[self.config["cx_method"]], **self.config["cx_kwargs"]
        )
        toolbox.register(
            "mutate",
            mut_methods[self.config["mut_method"]],
            **self.config["mut_kwargs"],
        )
        toolbox.register(
            "select",
            sel_methods[self.config["sel_method"]],
            **self.config["sel_kwargs"],
        )
        toolbox.register("evaluate", self.evaluate)
        return toolbox

    def evaluate(self, individual):
        """Run EvoMan and return fitness of individual.

        Takes an individual (array of weights, in the case of a NN), and runs a simulation of
        an EvoMan game with that individual as a controller, returning the fitness of the individual
        as a result."""
        f, p, e, t = self.env.play(pcont=np.array(individual))
        return f

    def optimise(self):
        """Train an algorithm to play EvoMan by using the DEAP framework.

        Runs for several generations, selecting and mating the best algorithms according to the
        methods specified to find an optimised solution. Returns the last population of algorithms
        as a result.
        """
        pop = self.toolbox.population(n=self.config["population_size"])

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = [fit]

        for g in range(self.config["n_generations"]):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config["cx_probability"]:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutate individuals
            for mutant in offspring:
                if random.random() < self.config["mut_probability"]:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = [fit]

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            self.log_gen(g, [ind.fitness.values[0] for ind in pop])
        return pop

    def log_gen(self, n_gen, historical_results):
        print(f"Mean Fitness for Generation {n_gen}: {np.mean(historical_results)}.")
        with open(
            f'./{self.config["experiment_name"]}/fitnesses.json', "a"
        ) as out_file:
            out_file.write(
                json.dumps({"generation": n_gen, "fitnesses": historical_results})
            )
            out_file.write("\n")
