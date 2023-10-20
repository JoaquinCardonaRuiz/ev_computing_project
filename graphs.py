import json
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import font_manager as font_manager
from deap_algorithm import Evaluator
import pandas as pd

def load_data_lineplots(models, models_dirname, enemies, runs):
    data = {}
    for m in models: 
        data[m] = {}
        for e in enemies:
            data[m][e] = {}
            for r in runs:
                with open(f'experiment_results_generalist/{m}/e{e}/{models_dirname[m]}_e_{e}_r{r}/fitnesses.json','r') as file:
                    fit_data = []
                    while True:
                        try:
                            fit_data.append(json.loads(file.readline()))
                        except:
                            break
                with open(f'experiment_results_generalist/{m}/e{e}/{models_dirname[m]}_e_{e}_r{r}/results.json', 'r') as file:
                    res_data = json.loads(file.read())
                data[m][e][r] = {'means':[np.mean(gen['fitnesses']) for gen in fit_data]+[res_data['mean']],
                                 'max':  [np.max(gen['fitnesses']) for gen in fit_data]+[res_data['max']]}

    new_dict = {}
    # Iterate through the original dictionary
    for model, enemies in data.items():
        for enemy, runs in enemies.items():
            for run, values in runs.items():
                means = values["means"]
                max_values = values["max"]
                for generation, (mean_value, max_value) in enumerate(zip(means, max_values)):
                    # Create keys in the new dictionary if not already present
                    if enemy not in new_dict:
                        new_dict[enemy] = {}
                    if model not in new_dict[enemy]:
                        new_dict[enemy][model] = {}
                    if f"{generation}" not in new_dict[enemy][model]:
                        new_dict[enemy][model][f"{generation}"] = {
                            "mean": [],
                            "max": []
                        }
                    # Append data to the new dictionary
                    new_dict[enemy][model][f"{generation}"]["mean"].append(mean_value)
                    new_dict[enemy][model][f"{generation}"]["max"].append(max_value)
    return new_dict

def lineplots():
    #sns.set_theme(font='Computer Modern Serif')
    models = ["Model 1", "Model 2"]
    models_colours = {"Model 1": "lightsteelblue", "Model 2": "midnightblue"}
    models_dirname = {"Model 1": "generalist_naa", "Model 2": "generalist_aa"}
    enemies = ["set_1","set_2"]
    enemy_names = {"set_1":'[2,5]',"set_2":'[2,4,6,7]'}
    runs = [str(i) for i in range(10)]
    generations = list(range(76))
    data = load_data_lineplots(models, models_dirname, enemies, runs)

    for e in enemies:
        for m in models:
            data_xgens_this_model = {'avg': np.array([np.mean(data[e][m][f'{gen}']['mean']) for gen in generations]), 
                                     'max': np.array([np.mean(data[e][m][f'{gen}']['max']) for gen in generations]), 
                                     'std_avg': np.array([np.std(data[e][m][f'{gen}']['mean']) for gen in generations]), 
                                     'std_max': np.array([np.std(data[e][m][f'{gen}']['max']) for gen in generations])}
            plt.plot(generations, data_xgens_this_model['avg'], '-', color = models_colours[m])
            plt.plot(generations, data_xgens_this_model['max'], '--', color = models_colours[m])
            plt.fill_between(generations, data_xgens_this_model['avg']-data_xgens_this_model['std_avg'], data_xgens_this_model['avg']+data_xgens_this_model['std_avg'], color = models_colours[m], alpha=0.3, label='_nolegend_')
            plt.fill_between(generations, data_xgens_this_model['max']-data_xgens_this_model['std_max'], data_xgens_this_model['max']+data_xgens_this_model['std_max'], color = models_colours[m], alpha=0.3, label='_nolegend_')
        plt.grid(zorder=0, color='lightgrey', alpha=0.7)
        plt.xticks(np.arange(0,76,5))
        plt.yticks(np.arange(0,100,5))
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.legend(["Model 1 (No Adaptive α) mean", "Model 1 (No Adaptive α) max", "Model 2 (Adaptive α) mean", "Model 2 (Adaptive α) max"], loc='lower right')
        plt.title(f'Fitness over Generations for Enemies {enemy_names[e]}')
        plt.show()

def param_tuning_graphs():
    exps = ['frank_params_hm', 'joaquin_params_hm', 'joaquin_params_lm', 'joaquin_params_mm', 'matt_params_hm', 'matt_params_lm', 'arina_params']
    ens = ['1', '2', '4', '5', '7']

    width = 0.2
    
    data = []
    for e in ens:
        enemy_data = []
        for exp in exps:
            with open(f'{exp}_e{e}/results.json', 'r') as file:
                json_data = json.loads(file.read())
            enemy_data.append(json_data['std'])
        data.append(enemy_data)

    x = np.arange(7)*2
    fig, ax = plt.subplots(1, 1)
    ax.grid(zorder=0, color='lightgrey', )
    ax.bar(x-0.4, data[0], width, color='darkseagreen', zorder=3)
    ax.bar(x-0.2, data[1], width, color='mediumseagreen', zorder=3)
    ax.bar(x, data[2], width, color='seagreen', zorder=3)
    ax.bar(x+0.2, data[3], width, color='green', zorder=3)
    ax.bar(x+0.4, data[4], width, color='darkgreen', zorder=3)
    plt.xticks(x, ['Frank', 'Joaquin hm', 'Joaquin lm', 'Joaquin mm', 'Matt hm', 'Matt lm', 'Arina'])
    plt.yticks(np.arange(0,15,0.5))
    plt.xlabel("Experiment")
    plt.ylabel("Fitness")
    plt.legend(["Enemy 1", "Enemy 2", "Enemy 4", "Enemy 5", "Enemy 7"], loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.show()

def load_data_boxplot(models, models_dirname, enemies, runs):
    data = {}
    for m in models: 
        data[m] = {}
        for e in enemies:
            data[m][e] = {}
            for r in runs:
                with open(f'experiment_results_generalist/{m}/e{e}/{models_dirname[m]}_e_{e}_r{r}/fitnesses.json','r') as file:
                    fit_data = []
                    while True:
                        try:
                            fit_data.append(json.loads(file.readline()))
                        except:
                            break
                with open(f'experiment_results_generalist/{m}/e{e}/{models_dirname[m]}_e_{e}_r{r}/results.json', 'r') as file:
                    res_data = json.loads(file.read())
                data[m][e][r] = res_data['best']

    return data

def load_data_diversity(models, models_dirname, enemies, runs):
    data = {}
    for m in models: 
        data[m] = {}
        for e in enemies:
            data[m][e] = {}
            for r in runs:
                with open(f'experiment_results_generalist/{m}/e{e}/{models_dirname[m]}_e_{e}_r{r}/fitnesses.json','r') as file:
                    fit_data = []
                    while True:
                        try:
                            fit_data.append(json.loads(file.readline()))
                        except:
                            break
                with open(f'experiment_results_generalist/{m}/e{e}/{models_dirname[m]}_e_{e}_r{r}/results.json', 'r') as file:
                    res_data = json.loads(file.read())
                data[m][e][r] = {'shannon': [gen['diversity_shannon'] for gen in fit_data]}

    new_dict = {}
    # Iterate through the original dictionary
    for model, enemies in data.items():
        for enemy, runs in enemies.items():
            for run, values in runs.items():
                shannon = values["shannon"]
                for generation, shannon_value in enumerate(shannon):
                    # Create keys in the new dictionary if not already present
                    if enemy not in new_dict:
                        new_dict[enemy] = {}
                    if model not in new_dict[enemy]:
                        new_dict[enemy][model] = {}
                    if f"{generation}" not in new_dict[enemy][model]:
                        new_dict[enemy][model][f"{generation}"] = {
                            "shannon": []
                        }
                    # Append data to the new dictionary
                    new_dict[enemy][model][f"{generation}"]["shannon"].append(shannon_value)
    return new_dict

def diversity():
    models = ["Model 1", "Model 2"]
    models_colours = {"Model 1": "lightsteelblue", "Model 2": "midnightblue"}
    enemies_dashes = {"set_1": "-", "set_2": "--"}
    models_dirname = {"Model 1": "generalist_naa", "Model 2": "generalist_aa"}
    enemy_names = {"set_1":'[2,5]',"set_2":'[2,4,6,7]'}
    enemies = ["set_1","set_2"]
    runs = [str(i) for i in range(10)]
    generations = list(range(75))
    data = load_data_diversity(models, models_dirname, enemies, runs)
    legends = []
    for e in enemies:
        for m in models:
            data_xgens_this_model = np.array([np.mean(data[e][m][f'{gen}']['shannon']) for gen in generations])
            legends.append(f'{m} | Enemy {enemy_names[e]}')
            plt.plot(generations, data_xgens_this_model, enemies_dashes[e], color = models_colours[m])
    plt.grid(zorder=0, color='lightgrey', alpha=0.7)
    plt.xticks(np.arange(0,75,5))
    plt.yticks(np.arange(0,3.25,0.25))
    plt.xlabel("Generations")
    plt.ylabel("Diversity")
    plt.legend(legends, loc='upper right')
    plt.title(f'Diversity by Model and Enemy')
    plt.show()

def eval_table_data():
    models = ["Model 1", "Model 2"]
    models_dirname = {"Model 1": "generalist_naa", "Model 2": "generalist_aa"}
    enemies = ["set_1","set_2"]
    runs = [str(i) for i in range(10)]
    data = load_data_boxplot(models, models_dirname,enemies,runs)
    results = {}
    for m in models:
        for e in enemies:
            results[f'run_{m}_e_{e}'] = []
            for r in runs:
                gains = []
                ev = Evaluator({'run_name': f'run_{m}_e_{e}_r{r}', 
                                'enemies': [1,2,3,4,5,6,7,8],
                                'h_neurons': 10,
                                'weights': data[m][e][r]})
                for i in range(5):
                    g = ev.run()
                    gains.append(g)
                results[f'run_{m}_e_{e}'].append(np.mean(gains))
    with open('boxplot_results.json', 'a') as file:
        file.write(json.dumps(results))

def table():
    pass

def eval_data():
    models = ["Model 1", "Model 2"]
    models_dirname = {"Model 1": "generalist_naa", "Model 2": "generalist_aa"}
    enemies = ["set_1","set_2"]
    runs = [str(i) for i in range(10)]
    data = load_data_boxplot(models, models_dirname,enemies,runs)
    results = {}
    for m in models:
        for e in enemies:
            results[f'run_{m}_e_{e}'] = []
            for r in runs:
                gains = []
                ev = Evaluator({'run_name': f'run_{m}_e_{e}_r{r}', 
                                'enemies': [1,2,3,4,5,6,7,8],
                                'h_neurons': 10,
                                'weights': data[m][e][r]})
                for i in range(5):
                    g = ev.run()
                    gains.append(g)
                results[f'run_{m}_e_{e}'].append(np.mean(gains))
    with open('boxplot_results.json', 'a') as file:
        file.write(json.dumps(results))
                

def boxplot():
    sns.set_style(style='white')
    models_colours = {"Model 1": "lightsteelblue", "Model 2": "midnightblue"}
    with open('boxplot_results.json','r') as file:
        data = json.loads(file.read())
    df = pd.DataFrame(data)
    df = df[['run_Model 1_e_set_1', 'run_Model 2_e_set_1', 'run_Model 1_e_set_2', 'run_Model 2_e_set_2']]
    df = df.rename(columns={'run_Model 1_e_set_1': 'Model 1 \n(No Adaptive α) \n Trained on enemies [2,5]',
                            'run_Model 1_e_set_2': 'Model 1 \n(No Adaptive α) \n Trained on enemies [2,4,6,7]',
                            'run_Model 2_e_set_1': 'Model 2 \n(Adaptive α) \n Trained on enemies [2,5]',
                            'run_Model 2_e_set_2': 'Model 2 \n(Adaptive α) \n Trained on enemies [2,4,6,7]'})
    df = df.melt() 
    df['model'] = df['variable'].str[:7]
    df = df.rename(columns={'variable': 'Experiment', 'value': 'Gain', 'model': 'Model'})
    plt.grid(color='lightgrey')
    plt.xticks(fontsize=10)
    plt.yticks(np.arange(-800,800,25))
    plt.ylabel('Individual Gain')
    plt.xlabel('',color="white")
    plt.title('Gain of Best Individuals Across 10 Rounds\nAveraged across 5 evaluations across all 8 enemies')
    sns.boxplot(x=df['Experiment'], y=df['Gain'],hue=df['Model'],palette=list(models_colours.values()), medianprops=dict(color="darkgrey"))
    plt.show()

diversity()
lineplots()
#eval_data()
#boxplot()

#table()