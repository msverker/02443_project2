import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math 
import tqdm
import scipy
import pandas as pd 
import sys

"""
0: healthy 
1: infected
2: recovered

3: vaccine
4: vaccine + infected
5: vaccine + recovered
"""

# from Hugo 
beta = 0.132 
gamma = 0.073

N = 100_000
initial_sick = 100
number_of_days = 500

population_list = np.array(
    [0] * (N - initial_sick) + [1] * initial_sick
)
sick_list = [0] * (N-initial_sick) + list(np.random.exponential(scale=1/gamma, size=initial_sick))
sick_list = [(0, v) for v in sick_list] # save day and time seperately

susceptible_list = [(N - initial_sick) / N]
infected_list = [initial_sick / N]
infected_vaccine_list = [0]
recovered_list = [0]
recovered_vaccine_list = [0]
vaccine_list = [0]

vaccinations_per_day = 100
if len(sys.argv) > 1:
    vaccine_effectiveness = float(sys.argv[1])
else:
    vaccine_effectiveness = 0.9

for day in tqdm.tqdm(range(number_of_days)):
    # Vaccination 
    can_be_vaccinated = np.array(range(N))[population_list == 0]
    vaccinated = np.random.choice(
        a=can_be_vaccinated,
        size=min(len(can_be_vaccinated), vaccinations_per_day), 
        replace=False
    )
    for person in vaccinated:
        population_list[person] = 3

    # New infections 
    S0 = np.array(range(N))[population_list == 0]
    S3 = np.array(range(N))[population_list == 3]
    I = np.array(range(N))[(population_list == 1) | (population_list == 4)]

    p_infection = beta * len(I) / N
    to_be_infected_0 = S0[np.random.random(size=len(S0)) < p_infection]
    to_be_infected_3 = S3[np.random.random(size=len(S3)) < p_infection * (1 - vaccine_effectiveness)]

    ## Assuming vaccinated people are as sick as non-vaccinated
    for person in np.hstack((to_be_infected_0, to_be_infected_3)):
        if population_list[person] == 0: population_list[person] = 1
        elif population_list[person] == 3: population_list[person] = 4

        sick_list[person] = (day, np.random.exponential(scale=1/gamma))

    # New recovered
    for person in I:
        infection_day, sick_time = sick_list[person]
        if day > infection_day + sick_time:
            if population_list[person] == 1: population_list[person] = 2
            elif population_list[person] == 4: population_list[person] = 5

    # Update lists 
    susceptible_list.append(
        np.sum((population_list == 0) | (population_list == 3)) / N
    )
    infected_list.append(
        np.sum(population_list == 1) / N
    )
    infected_vaccine_list.append(
        np.sum(population_list == 4) / N
    )
    recovered_list.append(
        np.sum(population_list == 2) / N 
    )
    recovered_vaccine_list.append(
        np.sum(population_list == 5) / N
    )
    vaccine_list.append(
        np.sum((population_list == 3) | (population_list == 4) | (population_list == 5)) / N
    )

plt.figure(figsize=(10, 6))
plt.title(f"Vaccine effectiveness={vaccine_effectiveness}")
plt.plot(susceptible_list, color="blue", label="Susceptible")
plt.plot(infected_list, color="red", label="Infected")
plt.plot(infected_vaccine_list, color="green", label="Infected vaccine")
plt.plot(recovered_list, color="black", label="Recovered")
plt.plot(recovered_vaccine_list, color="orange", label="Recovered vaccine")
plt.plot(vaccine_list, color="magenta", label="Vaccine")

plt.grid(visible=True)
plt.legend()

plt.savefig(f"vaccine_model_effectiveness_{vaccine_effectiveness}.png")