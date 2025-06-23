import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math 
import tqdm
import scipy
import pandas as pd 

# from Hugo 
beta = 0.132 
gamma = 0.073

N = 100_000
initial_sick = 100
number_of_days = 400

"""
0: healthy
1: infected
2: immune for some time and then back to healthy (0)
"""

beta = 0.132 
gamma = 0.073

N = 100_000
initial_sick = 100 
number_of_days = 2_000
immune_time = 120

population_list = np.array(
    [0] * (N - initial_sick) + [1] * initial_sick
)
sick_list = [0] * (N-initial_sick) + list(np.random.exponential(scale=1/gamma, size=initial_sick))
sick_list = [(0, v) for v in sick_list]
immune_list_time = [(0, 0)] * N

susceptible_list = [(N - initial_sick) / N]
infected_list = [initial_sick / N]
immune_list = [0]

for day in tqdm.tqdm(range(number_of_days)):
    S = np.array(range(N))[population_list == 0]
    I = np.array(range(N))[population_list == 1]
    IM = np.array(range(N))[population_list == 2]
    
    # New infected (0 to 1)
    p_infection = beta * len(I) / N
    to_be_infected = S[np.random.random(size=len(S)) < p_infection]
    for person in to_be_infected:
        population_list[person] = 1
        sick_list[person] = (day, np.random.exponential(scale=1/gamma))

    # New immune (1 to 2)
    for person in I:
        infection_day, sick_time = sick_list[person]
        if day > infection_day + sick_time:
            population_list[person] = 2
            immune_list_time[person] = (day, np.random.exponential(scale=immune_time))

    # New healthy (2 to 0)
    for person in IM:
        immune_day, immune_time = immune_list_time[person]
        if day > immune_day + immune_time:
            population_list[person] = 0

    # Update the proposition lists
    susceptible_list.append(np.sum(population_list == 0) / N)
    infected_list.append(np.sum(population_list == 1) / N)
    immune_list.append(np.sum(population_list == 2) / N)

plt.figure(figsize=(10, 6))
plt.grid(visible=True)
plt.title("S-I-IM model")
plt.plot(susceptible_list, label="Susceptible")
plt.plot(infected_list, label="Infected")
plt.plot(immune_list, label="Immune")

plt.legend()

plt.savefig("infected_after_some_time_model.png")