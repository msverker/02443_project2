import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def run_simulation(beta: float = 0.132,
                   gamma: float = 0.073,
                   N: int = 100_000,
                   initial_sick: int = 100,
                   number_of_days: int = 2_000,
                   immune_time: int = 120):

    population = np.zeros(N, dtype=np.uint8)
    population[-initial_sick:] = 1

    sick_times = np.zeros(N)
    sick_times[-initial_sick:] = np.random.exponential(scale=1/gamma, size=initial_sick)
    sick_list = np.zeros((N, 2))
    sick_list[-initial_sick:, 1] = sick_times[-initial_sick:]

    immune_list_time = np.zeros((N, 2))

    has_been_infected = np.zeros(N, dtype=bool)
    has_been_infected[-initial_sick:] = True

    time_since_last_infected = np.zeros(N, dtype=int)
    time_since_last_infected_list = []

    susceptible_list = [(N - initial_sick) / N]
    infected_list = [initial_sick / N]
    immune_list = [0.0]

    for day in tqdm(range(number_of_days), disable=True):
        S = np.where(population == 0)[0]
        I = np.where(population == 1)[0]
        IM = np.where(population == 2)[0]

        mask = has_been_infected[S]
        time_since_last_infected[S[mask]] += 1

        if len(I) > 0:
            p_infection = beta * len(I) / N
            infected_now = S[np.random.random(size=len(S)) < p_infection]
            population[infected_now] = 1
            durations = np.random.exponential(scale=1/gamma, size=len(infected_now))
            sick_list[infected_now, 0] = day
            sick_list[infected_now, 1] = durations
            has_been_infected[infected_now] = True

            infected_intervals = time_since_last_infected[infected_now]
            reinfected = infected_intervals > 0
            time_since_last_infected_list.extend(infected_intervals[reinfected])
            time_since_last_infected[infected_now[reinfected]] = 0

        # Recovery (1 → 2)
        recovery_mask = (day > (sick_list[:, 0] + sick_list[:, 1])) & (population == 1)
        recovery_indices = np.where(recovery_mask)[0]
        population[recovery_indices] = 2
        immune_list_time[recovery_indices, 0] = day
        immune_list_time[recovery_indices, 1] = np.random.exponential(scale=immune_time, size=len(recovery_indices))

        # Immunity wanes (2 → 0)
        immunity_waned_mask = (day > (immune_list_time[:, 0] + immune_list_time[:, 1])) & (population == 2)
        population[immunity_waned_mask] = 0

        # Track population proportions
        susceptible_list.append(np.mean(population == 0))
        infected_list.append(np.mean(population == 1))
        immune_list.append(np.mean(population == 2))

    return susceptible_list, infected_list, immune_list, time_since_last_infected_list

def run_ci(immune_time):
    num_runs = 10
    mean_reinfection_times = []

    for _ in range(num_runs):
        _, _, _, reinfection_intervals = run_simulation(immune_time=immune_time)
        if reinfection_intervals:
            mean_reinfection_times.append(np.mean(reinfection_intervals))

    data = np.array(mean_reinfection_times)
    overall_mean = np.mean(data)
    std_err = stats.sem(data)
    confidence = 0.95
    h = std_err * stats.t.ppf((1 + confidence) / 2., len(data) - 1)

    print(f"Mean reinfection interval over {num_runs} runs: {overall_mean:.2f} using immune time {immune_time} days")
    print(f"95% Confidence Interval: ({overall_mean - h:.2f}, {overall_mean + h:.2f})")

def plot_sim(save=False):
    susceptible_list, infected_list, immune_list, _ = run_simulation()
    plt.figure(figsize=(10, 6))
    plt.grid(visible=True)
    plt.title("S-I-IM model")
    plt.plot(susceptible_list, label="Susceptible")
    plt.plot(infected_list, label="Infected")
    plt.plot(immune_list, label="Immune")

    plt.legend()
    if save:
        plt.savefig("infected_after_some_time_model.png")
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for modelling infection after time.")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--immune_time", type=int, default=120)
        
    args = parser.parse_args()

    if args.ci:
        run_ci(args.immune_time)
    if args.plot:
        plot_sim()

    