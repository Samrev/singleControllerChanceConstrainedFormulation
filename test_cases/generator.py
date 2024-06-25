import random
import sys
import numpy as np
import scipy.stats as stats
from sklearn.datasets import make_spd_matrix

# N -> number of states
N = int(sys.argv[1])

# Define the range for the parameters
p_min = 0
p_max = 5
c_min = 0
c_max = 15
max_actions1 = 5
max_actions2 = 5
v_min = 0
v_max = 5
beta = 0.75

def check(num):
    if num == 0:
        return False
    while (num % 2) == 0:
        num = num // 2
    while (num % 5) == 0:
        num = num // 5
    return num == 1

# Generate random values for P, C1, and C2
S = set(range(1, N + 1))
A1 = {s: set(range(1, random.randint(1, max_actions1) + 1)) for s in S}
A2 = {s: set(range(1, random.randint(1, max_actions2) + 1)) for s in S}
P = {}
A1 = {1:{1} , 2:{1,2}}
A2 = {1:{1,2}, 2:{1}}
gamma = {}

for s in S:
    for a2 in A2[s]:
        sum_probabilities = 0
        while not check(sum_probabilities):
            temp = {(s_, s, a2): random.randint(p_min, p_max) for s_ in S}
            sum_probabilities = sum(temp[s_, s, a2] for s_ in S)
        P.update({key: value / sum_probabilities for key, value in temp.items()})


# Initial distribution
for s in S:
    sum_probabilities = 0
    while not check(sum_probabilities):
        temp = {s: random.randint(p_min, p_max) for s in S}
        sum_probabilities = sum(temp[s] for s in S)
    gamma.update({key: value / sum_probabilities for key, value in temp.items()})
gamma={1:0.5,2:0.5}
# Generate mu and sigmas
dimension = sum(len(A1[s]) * len(A2[s]) for s in S)

mu1 = np.random.randint(c_min, c_max, dimension)
sigma1 = make_spd_matrix(n_dim=dimension, random_state=42)
mu2 = np.random.randint(c_min, c_max, dimension)
sigma2 = make_spd_matrix(n_dim=dimension, random_state=42)

sigma1 = np.array([
    [10, 2, 3, 4],
    [2, 8, 2, 1],
    [3, 2, 9, 2],
    [4, 1, 2, 7]
])
sigma2 = np.array([
    [6, 2, 1, 3],
    [2, 5, 2, 1],
    [1, 2, 4, 2],
    [3, 1, 2, 6]
])
# Generate alphas
alpha1 = random.uniform(0.5, 1)
alpha2 = random.uniform(0.5, 1)
F_inv_alpha1 = round(stats.norm.ppf(alpha1),2)
F_inv_alpha2 = round(stats.norm.ppf(alpha2),2)
F_inv_alpha1 = 1
F_inv_alpha2 = 1
# Write the data to a file
with open(f'{N}stdata.dat', 'w') as data_file:
    data_file.write("data;\n")
    data_file.write(f"set S := {', '.join(map(str, S))};\n")
    for s in S:
        data_file.write(f"set A1[{s}] := {', '.join(map(str, A1[s]))};\n")
        data_file.write(f"set A2[{s}] := {', '.join(map(str, A2[s]))};\n")
    
    data_file.write("param P := ")
    for (s_, s, a2) in P:
        data_file.write(f"\n  {s_} {s} {a2} {P[s_, s, a2]}")
    data_file.write(";\n")

    data_file.write(f"param F_inv_alph1 := {F_inv_alpha1};\n")
    data_file.write(f"param F_inv_alph2 := {F_inv_alpha2};\n")

    data_file.write(f"param gamma := ")
    for s in S:
        data_file.write(f"\n {s} {gamma[s]}")
    data_file.write(";\n")

    data_file.write(f"param mu1 :=")
    curr = 0
    for s in range(1, N + 1):
        for a1 in range(1, len(A1[s]) + 1):
            for a2 in range(1, len(A2[s]) + 1):
                data_file.write(f"\n {s} {a1} {a2} {mu1[curr]}")
                curr += 1
    data_file.write(";\n")

    data_file.write(f"param mu2 :=")
    curr = 0
    for s in range(1, N + 1):
        for a1 in range(1, len(A1[s]) + 1):
            for a2 in range(1, len(A2[s]) + 1):
                data_file.write(f"\n {s} {a1} {a2} {mu2[curr]}")
                curr += 1
    data_file.write(";\n")

    data_file.write(f"param sigma1 :=")
    curr1 = 0
    for s in range(1, N + 1):
        for a1 in range(1, len(A1[s]) + 1):
            for a2 in range(1, len(A2[s]) + 1):
                curr2 = 0
                for s_ in range(1, N + 1):
                    for a1_ in range(1, len(A1[s_]) + 1):
                        for a2_ in range(1, len(A2[s_]) + 1):
                            data_file.write(f"\n {s} {a1} {a2} {s_} {a1_} {a2_} {sigma1[curr1, curr2]}")
                            curr2 += 1
                curr1 += 1
    data_file.write(";\n")

    data_file.write(f"param sigma2 :=")
    curr1 = 0
    for s in range(1, N + 1):
        for a1 in range(1, len(A1[s]) + 1):
            for a2 in range(1, len(A2[s]) + 1):
                curr2 = 0
                for s_ in range(1, N + 1):
                    for a1_ in range(1, len(A1[s_]) + 1):
                        for a2_ in range(1, len(A2[s_]) + 1):
                            data_file.write(f"\n {s} {a1} {a2} {s_} {a1_} {a2_} {sigma2[curr1, curr2]}")
                            curr2 += 1
                curr1 += 1
    data_file.write(";\n")
