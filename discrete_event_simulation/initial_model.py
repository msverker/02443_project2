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

population_list = np.array(
    [0] * (N - initial_sick) + [1] * initial_sick
)
sick_list = [0] * (N-initial_sick) + list(np.random.exponential(scale=1/gamma, size=initial_sick))
sick_list = [(0, v) for v in sick_list] # save day and time seperately

susceptible_list = [(N - initial_sick) / N]
infected_list = [initial_sick / N]
recovered_list = [0]

for day in tqdm.tqdm(range(number_of_days)):
    S = np.array(range(N))[population_list == 0]
    I = np.array(range(N))[population_list == 1]

    # New infections (0 to 1)
    p_infection = beta * len(I) / N
    to_be_infected = S[np.random.random(size=len(S)) < p_infection]
    for person in to_be_infected:
        population_list[person] = 1
        sick_list[person] = (day, np.random.exponential(scale=1/gamma))

    # New healthy (1 to 2)
    for person in I:
        infection_day, sick_time = sick_list[person]
        if day > infection_day + sick_time:
            population_list[person] = 2

    # Update the proposition lists
    susceptible_list.append(np.sum(population_list == 0) / N)
    infected_list.append(np.sum(population_list == 1) / N)
    recovered_list.append(np.sum(population_list == 2) / N)

plt.figure(figsize=(10, 6))
plt.title("Initial SIR model")
plt.plot(susceptible_list, label="Susceptible")
plt.plot(infected_list, label="Infected")
plt.plot(recovered_list, label="Recovered")

plt.legend()
plt.savefig("initial_model.png")