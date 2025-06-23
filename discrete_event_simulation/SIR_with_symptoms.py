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

incubation_time = 4 # ???

# No symptoms does not infect
population_list = np.array(
        [0] * (N - initial_sick) + [2] * initial_sick
    )
sick_list_time = [0] * (N-initial_sick) + list(np.random.exponential(scale=1/gamma, size=initial_sick))
sick_list_time = [(0, v) for v in sick_list_time] # save day and time seperately
no_symptoms_time = [(0, 0)] * N 

susceptible_list = [(N - initial_sick) / N]
infected_list = [initial_sick / N]
recovered_list = [0]
no_symptoms_list = [0]

for day in tqdm.tqdm(range(number_of_days)):
    S = np.array(range(N))[population_list == 0]
    I = np.array(range(N))[population_list == 2]
    
    # Infected people (2) makes healthy (0) become no symptom (1)
    p_infection = beta * len(I) / N
    to_be_infected = S[np.random.random(size=len(S)) < p_infection]
    for person in to_be_infected:
        population_list[person] = 1
        no_symptoms_time[person] = (day, np.random.exponential(scale=incubation_time)) 

    # No symptoms (1) become infected (2)
    NS = np.array(range(N))[population_list == 1]
    for person in NS:
        infection_day, infection_time = no_symptoms_time[person]
        if day > infection_day + infection_time:
            population_list[person] = 2
            sick_list_time[person] = (day, np.random.exponential(scale=1/gamma))

    # Infected (2) recoveres (3)
    for person in I:
        sick_day, sick_time = sick_list_time[person]
        if day > sick_day + sick_time:
            population_list[person] = 3

    # Update the proposition lists        
    susceptible_list.append(
        np.sum(population_list == 0) / N
    )
    no_symptoms_list.append(
        np.sum(population_list == 1) / N
    )
    infected_list.append(
        np.sum(population_list == 2) / N
    )
    recovered_list.append(
        np.sum(population_list == 3) / N 
    )

plt.figure(figsize=(10, 6))
plt.title("SSIR")
plt.grid(visible=True)

plt.plot(susceptible_list, color="blue", label="Susceptible")
plt.plot(no_symptoms_list, color="black", label="No symptoms")
plt.plot(infected_list, color="red", label="Infected")
plt.plot(recovered_list, color="orange", label="Recovered")

plt.legend()
plt.savefig("SIR_with_symptoms.png")

# No symptoms infects people
population_list = np.array(
        [0] * (N - initial_sick) + [2] * initial_sick
    )
sick_list_time = [0] * (N-initial_sick) + list(np.random.exponential(scale=1/gamma, size=initial_sick))
sick_list_time = [(0, v) for v in sick_list_time] # save day and time seperately
no_symptoms_time = [(0, 0)] * N 

susceptible_list = [(N - initial_sick) / N]
infected_list = [initial_sick / N]
recovered_list = [0]
no_symptoms_list = [0]

for day in tqdm.tqdm(range(number_of_days)):
    S = np.array(range(N))[population_list == 0]
    I2 = np.array(range(N))[population_list == 2]
    I1 = np.array(range(N))[population_list == 1]
    I = np.hstack((I1, I2))
    
    # Infected people (1 or 2) makes healthy (0) become no symptom (1)
    p_infection = beta * len(I) / N
    to_be_infected = S[np.random.random(size=len(S)) < p_infection]
    for person in to_be_infected:
        population_list[person] = 1
        no_symptoms_time[person] = (day, np.random.exponential(scale=incubation_time)) 

    # No symptoms (1) become infected (2)
    NS = np.array(range(N))[population_list == 1]
    for person in NS:
        infection_day, infection_time = no_symptoms_time[person]
        if day > infection_day + infection_time:
            population_list[person] = 2
            sick_list_time[person] = (day, np.random.exponential(scale=1/gamma))

    # Infected (2) recoveres (3)
    for person in I2:
        sick_day, sick_time = sick_list_time[person]
        if day > sick_day + sick_time:
            population_list[person] = 3

    # Update the proposition lists        
    susceptible_list.append(
        np.sum(population_list == 0) / N
    )
    no_symptoms_list.append(
        np.sum(population_list == 1) / N
    )
    infected_list.append(
        np.sum(population_list == 2) / N
    )
    recovered_list.append(
        np.sum(population_list == 3) / N 
    )

plt.figure(figsize=(10, 6))
plt.title("SSIR where no symptoms also infects")
plt.grid(visible=True)

plt.plot(susceptible_list, color="blue", label="Susceptible")
plt.plot(no_symptoms_list, color="black", label="No symptoms")
plt.plot(infected_list, color="red", label="Infected")
plt.plot(recovered_list, color="orange", label="Recovered")

plt.legend()
plt.savefig("SIR_with_symptoms_infects.png")
