import json
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

dirs = ['stability_test_final_aa_e_set_1_r0',
        'stability_test_final_aa_e_set_2_r0',
        'stability_test_final_naa_e_set_1_r0',
        'stability_test_final_naa_e_set_2_r0',
        ]

data = []
for dir in dirs:
    dir_data = []
    with open(f'{dir}/fitnesses.json','r') as file:
        while True:
            try:
                dir_data.append(np.mean(json.loads(file.readline())['fitnesses']))
            except:
                break
    data.append(dir_data)

print(data)
for d in data:
    sns.lineplot(d)
plt.show()