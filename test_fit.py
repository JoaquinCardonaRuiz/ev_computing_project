import random
import scipy
import math
from matplotlib import pyplot as plt
import numpy as np

a = np.random.randint(0,10, (5,3))
f = np.random.randint(0,100, (5))

r_fitnesses = np.zeros_like(f)

# Calculate Hamming distances for a chunk of individuals
dist_matrix = scipy.spatial.distance.pdist(a, metric="hamming")
dist_matrix = scipy.spatial.distance.squareform(dist_matrix)*3

# Calculate denominators
cond_matrix = dist_matrix <= 2
den_factors = cond_matrix * (1 - (np.power((dist_matrix / 2),1)))
denominators = np.sum(den_factors, axis=1)

# Calculate resulting fitnesses for the current chunk
r_fitnesses = f / denominators
print(a)
print(dist_matrix)
print(cond_matrix)
print(den_factors)
print(denominators)
print(f)
print(r_fitnesses)

quit()
#print(np.array(dist_matrix).shape)
print(a)
dist_matrix = np.round(np.linalg.norm(a[:, None, :] - a[None, :, :], axis=-1))
cond_matrix = dist_matrix <=7
den_factors = cond_matrix - cond_matrix*(np.power((dist_matrix/7),1))
den_factors = cond_matrix * (1 - (np.power((dist_matrix/7),2)))
denominators = np.sum(den_factors, axis=1)
print(dist_matrix)
print(cond_matrix)
print(den_factors)
print(denominators)
print("fitnesses: ")
print(f)
print(f/denominators)

s = [math.sin(i/100)*((i/3140)+0.1)+2 for i in range(3140)]

inds = [random.randint(0,3139) for _ in range(1000)]

radius = 50

s_fit = s[:]

for individual in range(len(inds)):
    denominator = 1
    for neighbour in range(len(inds)):
        if individual == neighbour:
            continue
        distance = abs(inds[individual]-inds[neighbour])
        if distance < radius:
            denominator += 1-(distance/radius)**0.005
    s_fit[inds[individual]] /= denominator


inds_new = []
inds_fit = []
for i,ind in enumerate(inds):
    if random.random() + 1.968 < s[ind] and random.random() + 1.968 < s[ind]:
        inds_new.append(ind)
    if random.random() + 1.55 < s_fit[ind] and random.random() + 1.55 < s_fit[ind]:
        inds_fit.append(ind)
print(len(inds_new))
print(len(inds_fit))

no_fit = plt.scatter(inds_new, [s[ind] for ind in inds_new])
fit = plt.scatter(inds_fit, [s_fit[ind] for ind in inds_fit])
plt.legend((no_fit, fit), ('No Fitness Sharing', 'Fitness Sharing'))
plt.show()
quit()

