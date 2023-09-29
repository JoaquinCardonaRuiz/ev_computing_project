import json
import numpy as np
from matplotlib import pyplot as plt

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