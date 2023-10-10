""" EvoMan project - DEAP code

This is the script that runs the EvoMan assignment for Evolutionary Computing using
the DEAP framework. It contains a class named DEAP_Optimiser, which needs to be initialised
with a JSON config string before the 'optimise' method can be called to run the algorithm.

The configuration JSON strings determines whether the algorithm returns or logs its results.
"""
# Built-in imports
import gc
import os
import json
import scipy
import random
import numpy as np
from operator import attrgetter
from matplotlib import pyplot as plt
from datetime import datetime

# DEAP imports
from deap import base, creator, tools
from deap.tools.selection import __all__ as all_selections
from deap.tools.mutation import __all__ as all_mutations
from deap.tools.crossover import __all__ as all_crossovers

# Evoman Imports
from evoman.environment import Environment
from demo_controller import player_controller


def selNonRepTournament(individuals, k, tournsize):
    """Select the best individual among *tournsize* randomly chosen
    individuals, *k* times. The list returned contains
    references to the input *individuals*.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param tournsize: The number of individuals participating in each tournament.
    :param fit_attr: The attribute of individuals to use as selection criterion
    :returns: A list of selected individuals.

    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    # Check that we won't run out of individuals to pick
    #if len(individuals) - k < tournsize: raise ValueError('Tournament: too few individuals for parameters specified.') 
    chosen = []
    for i in range(k):
        # remove individuals chosen from pool
        aspirants = np.random.permutation([i for i in range(len(individuals))])[:tournsize]
        chosen_one = max(aspirants, key=lambda x: individuals[x].fitness.values[0])
        chosen.append(individuals[chosen_one])
        del individuals[chosen_one]
    # add back individuals because otherwise they get removed from
    # original population, because everything in deap is referential
    individuals += chosen
    return chosen

class Evaluator():
    """ Class for evaluating best solutions for boxplot."""
    def __init__(self, config):
        self.config = config

        if not os.path.exists(self.config['run_name']):
            os.makedirs(self.config['run_name'])

        self.env = Environment(
            experiment_name=self.config['run_name'],
            enemies=[self.config['enemy']],
            playermode="ai",
            player_controller=player_controller(self.config['h_neurons']),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False)
        self.env.state_to_log()

    def run(self):
        f,p,e,t = self.env.play(pcont=np.array(self.config['weights']))
        return p-e
        
        

class DEAP_Optimiser():
    """ DEAP Optimiser class

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
        n_inputs, n_outputs  = 20, 5
        self.tot_neurons = self.config['h_neurons'] * (n_inputs + n_outputs + 1) + n_outputs

        # Set simulation Environment
        self.env = self.set_env()
        
        # Set up DEAP toolbox
        self.toolbox = self.set_toolbox()
      
    def set_env(self):
        """Initialises EvoMan simulation environment."""
        if not os.path.exists(self.config['experiment_name']):
            os.makedirs(self.config['experiment_name'])

        env = Environment(
            experiment_name=self.config['experiment_name'],
            enemies=[2,4,7,6],
            playermode="ai",
            multiplemode="yes",
            player_controller=player_controller(self.config['h_neurons']),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False)
        env.state_to_log()
        return env

    def set_toolbox(self):
        """Initialises DEAP framework with specified parameters."""
        toolbox = base.Toolbox()

        cx_methods =  {method_name:getattr(tools, method_name) for method_name in all_crossovers}
        mut_methods = {method_name:getattr(tools, method_name) for method_name in all_mutations}
        sel_methods = {method_name:getattr(tools, method_name) for method_name in all_selections}
        sel_methods['selNonRepTournament'] = selNonRepTournament

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax, 
                                           non_adj_fitness=creator.FitnessMax)
        toolbox.register("attr_float", np.random.uniform,-1,1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                              toolbox.attr_float, n=self.tot_neurons)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", cx_methods[self.config['cx_method']], **self.config['cx_kwargs'])
        toolbox.register("mutate", mut_methods[self.config['mut_method']], **self.config['mut_kwargs'])
        toolbox.register("parent_select", sel_methods[self.config['parent_sel_method']], **self.config['parent_sel_kwargs'])
        toolbox.register("survivor_select", sel_methods[self.config['survivor_sel_method']], **self.config['survivor_sel_kwargs'])
        toolbox.register("fitness_sharing", self.fitness_sharing_np)
        toolbox.register("measure_diversity", self.measure_diversity)
        toolbox.register("evaluate", self.evaluate)
        return toolbox
    
    def check_repeats(self, population, where):
        ids = [id(ind) for ind in population]
        vals = [list(ind) for ind in population]
        id_reps = any(ids.count(x) > 1 for x in ids)
        val_reps = any(vals.count(x) > 1 for x in vals)
        if id_reps and val_reps:
            print(f'{where}: Repeat by ids and vals!')
            _ = input('continue?')
        elif id_reps:
            print(f'{where}: Repeat ids only!')
            _ = input('continue?')
        elif val_reps:
            print(f'{where}: Repeat vals only!')
            _ = input('continue?')
        
    def evaluate(self, individual):
        """ Run EvoMan and return fitness of individual.
        
        Takes an individual (array of weights, in the case of a NN), and runs a simulation of
        an EvoMan game with that individual as a controller, returning the fitness of the individual
        as a result."""
        f,p,e,t = self.env.play(pcont=np.array(individual))
        return f

    def crossover(self, parents):
        """ Crossover subset of the population to generate offspring."""
        children = []
        for _ in range(self.config['children_per_parent']):
            # We repeat the suffling, assigning partners and crossover process
            # for every child we need per parent.
            np.random.shuffle(parents)
            for parent1, parent2 in zip(parents[::2], parents[1::2]):
                child1, child2 = self.toolbox.clone(parent1), self.toolbox.clone(parent2)
                self.toolbox.mate(child1, child2)
                children += [child1, child2]
                del child1.fitness.values
                del child2.fitness.values
        return children

    def eval_offspring(self, offspring):
        """Evaluate offpsring individuals.
        
        If any offpsring was present in previous generations, it will already
        have a fitness, and therefore get skipped here.
        """
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = [fit]
            ind.non_adj_fitness.values = [fit]

    def mutate(self, offspring):
        """Mutate offspring individuals."""
        for mutant in offspring:
            if random.random() < self.config['mut_probability']:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values


    def fitness_sharing_np(self, pop):
        """ Implements Fitness Sharing to preserve diversity in population

        Uses NumPy to calculate distances between networks. This is pretty
        bad code but it's the only way I managed to make it performant. 

        Applies the Frobenius matrix norm to the weights of the population
        to calculate distances and uses map operations to calculate new 
        fitness sharing factors.

        See https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm

        Fast(er) but very mem intensive.
        """
        # We use float16 to save memory. The first operation in this algorithm
        # involves creating a 3D matrix of size pop_size X pop_size X neurons
        # which can balloon to thousands of gigabytes if not careful
        weights = np.array([np.array(ind) for ind in pop],dtype=np.float16)
        fitnesses = np.array([ind.fitness.values[0] for ind in pop],dtype=np.float16)

        if self.config['dist_func'] == 'euclidean':
            # Get pairwise individual x individual distances using Frobenius norm
            dist_matrix = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=-1)
        elif self.config['dist_func'] == 'hamming':
            dist_matrix = scipy.spatial.distance.pdist(weights, metric="hamming")
            dist_matrix = scipy.spatial.distance.squareform(dist_matrix)*self.tot_neurons

        # Get boolean matrix with elements inside the radius
        cond_matrix = dist_matrix <= self.config['fitshare_radius']
        # Include strength factor
        den_factors = cond_matrix * (1 - (np.power((dist_matrix/ self.config['fitshare_radius']),self.config['fitshare_strength'])))
        # Clear memory
        del dist_matrix, cond_matrix, weights
        # Sum up denominators
        denominators = np.sum(den_factors, axis=1)
        # Get resulting fitnesses
        r_fitnesses = fitnesses/denominators
        # Assign new fitnesses
        for i in range(len(pop)):
            pop[i].fitness.values = [r_fitnesses[i]]

    def shannon_entropy(self, weights):
        weights = np.digitize(weights, bins=np.arange(np.min(weights), 
                                                    np.max(weights),
                                                    0.1))
        trans_weights = weights.transpose()
        count_weights = [np.bincount(row) for row in trans_weights]
        entropies = np.array([scipy.stats.entropy(row) for row in count_weights])
        return np.mean(entropies)

    def hamming_diversity(self, weights):
        dist_matrix = scipy.spatial.distance.pdist(weights, metric="hamming")
        dist_matrix = scipy.spatial.distance.squareform(dist_matrix)*self.tot_neurons
        return np.mean(dist_matrix)

    def measure_diversity(self, pop):
        """ Measure diversity using Shannon Entropy"""
        weights = np.array([np.array(ind) for ind in pop],dtype=np.float16)
        return({'shannon': self.shannon_entropy(weights), 
                'hamming': self.hamming_diversity(weights)})            

    def optimise(self):
        """ Train an algorithm to play EvoMan by using the DEAP framework.
        
        Runs for several generations, selecting and mating the best algorithms according to the
        methods specified to find an optimised solution. Returns the last population of algorithms 
        as a result.
        """
        pop = self.toolbox.population(n=self.config['population_size'])
        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = [fit]
            ind.non_adj_fitness.values = [fit]

        now  = datetime.now()
        for g in range(self.config['n_generations']):
            if g > 0:
                old_time = now
                now = datetime.now()
                print(f'Gen time: {(now-old_time).total_seconds()}s')
                
            self.log_gen(g, [ind.non_adj_fitness.values[0] for ind in pop], self.toolbox.measure_diversity(pop))

            # Parent selection | Note: select generates references to the individuals in pop.
            # To have all parents reproduce, select a 'parents' parameter equal to population
            # size, and a non-replacing selection method.
            parents_to_reproduce = self.toolbox.parent_select(pop, self.config['parents'])

            # Get offspring
            offspring = self.crossover(parents_to_reproduce)

            # Mutate and reevaluate
            self.mutate(offspring)
            self.eval_offspring(offspring)
            if self.config['do_fitshare']:
                self.toolbox.fitness_sharing(offspring)

            # Only delete references created by select, not actual parents. 
            # The parents still live in pop.
            del parents_to_reproduce
            
            if self.config['mode'] == 'steady':
                offspring = pop + offspring

            # We no longer need old population. If mode is generational, it's irrelevant.
            # If mode is steady-state, we already added it to the offspring population.
            del pop

            if len(offspring) < self.config['population_size']:
                # If this happens, it's because the mode was generational and the number of children
                # per parent was too low to replace population.
                raise ValueError(f'Optimise: parents produced too few offspring ({len(offspring)}/{self.config["population_size"]}), parameters not valid.')
            
            # Survivor selection | Note: select generates references to the individuals in offspring.
            # To have all children survive, choose non-replacing selection method
            pop = self.toolbox.survivor_select(offspring, self.config['population_size'])
            del offspring
        self.log_run(pop)
        return np.max([ind.non_adj_fitness.values[0] for ind in pop]) 


    def log_gen(self, n_gen, fitnesses, diversity):
        """Log fitnesses for current generation in disk."""
        print(f'Mean Fitness for Generation {n_gen}: {np.mean(fitnesses)}.')
        print(f'Mean Diversity for Generation {n_gen}: {diversity}.')
        with open(f'./{self.config["experiment_name"]}/fitnesses.json', 'a') as out_file:
            out_file.write(json.dumps({'generation':n_gen, 
                                       'fitnesses': fitnesses, 
                                       'diversity_shannon': diversity['shannon'], 
                                       'diversity_hamming': diversity['hamming']}))
            out_file.write("\n")

    def log_run(self, pop):
        print(f'====== Experiment {self.config["experiment_name"]} Finished ======')
        fitnesses = [ind.non_adj_fitness.values[0] for ind in pop]
        mean, mx, std = np.mean(fitnesses), np.max(fitnesses), np.std(fitnesses)
        best_ind = max(pop, key=attrgetter('fitness'))
        print(f'Resulting Mean Fitness: {mean}.')
        print(f'Resulting Max Fitness:  {mx}.')
        print(f'Resulting Std Fitness:  {std}.')
        with open(f'./{self.config["experiment_name"]}/results.json', 'a') as out_file:
            out_file.write(json.dumps({'mean':mean, 'max': mx, 'std': std, 'best': list(best_ind), 'config': self.config},indent=4))
            out_file.write("\n")



