This is group 18's (Four Robots) code repository for the specialist agent task in Evolutionary Computing.

The contents of the ./evoman directory, as well as all the files previously present in the provided repository, remain unchanged. The files added by the group are:

- run_experiments.py : **Run this file to run experiments**. This file loads the experiment configs from *deap_config.json* and runs the experiments with the EA built by the group.
- deap_config.json : **Set up experiments here**. This json file contains configs for experiments to be run. They must be loaded and fed to the EA as a python dict.
- deap_algorithm.py : Code for the EA built by the group (class DEAP_Optimiser), as well as an evaluator to re-evaluate found solutions (class Evaluator).
- param_serach.py : Runs dual-annealing parameter tuning, reading configs from *param_search_config.json*
- param_search_config.json : This json file contains configs for the parameter tuning.
- graphs.py : Code for building graphs based on recorded results.

The results of our experiments are stored in the directory named *experiment_results*. The results of the re-evaluations done for the box-plot are in *boxplot_results.json*. For this assignment we've chosen to use the default player controller in *demo_controller.py*, which is why the file is included.