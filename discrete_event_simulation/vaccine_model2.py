import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
import argparse


def run_simulation_with_vaccine(beta: float = 0.132,
                                gamma: float = 0.073,
                                N: int = 100_000,
                                initial_sick: int = 1_000,
                                number_of_days: int = 1_000,
                                immune_time: int = 90,
                                vaccinations_per_day: int = 100,
                                vaccine_effectiveness: float = 0.9):

    population = np.zeros(N, dtype=np.uint8)
    population[-initial_sick:] = 1

    sick_list_time = np.zeros((N, 2))
    sick_list_time[-initial_sick:, 1] = np.random.exponential(scale=1/gamma, size=initial_sick)

    immune_list_time = np.zeros((N, 2))

    susceptible_list = [(N - initial_sick) / N]
    infected_list = [initial_sick / N]
    infected_vaccine_list = [0]
    immune_list = [0]
    vaccine_list = [0]

    all_indices = np.arange(N)

    for day in tqdm(range(number_of_days)):
        # Vaccination (0 → 3, 2 → 3)
        can_be_vaccinated = all_indices[(population == 0) | (population == 2)]
        if len(can_be_vaccinated) > 0:
            vaccinated = np.random.choice(can_be_vaccinated,
                                          size=min(vaccinations_per_day, len(can_be_vaccinated)),
                                          replace=False)
            population[vaccinated] = 3

        # Infection
        S0 = all_indices[population == 0]
        S3 = all_indices[population == 3]
        I = all_indices[(population == 1) | (population == 4)]

        if len(I) > 0:
            p_infection = beta * len(I) / N
            to_be_infected_0 = S0[np.random.random(size=len(S0)) < p_infection]
            to_be_infected_3 = S3[np.random.random(size=len(S3)) < p_infection * (1 - vaccine_effectiveness)]

            for person in np.concatenate((to_be_infected_0, to_be_infected_3)):
                if population[person] == 0:
                    population[person] = 1
                elif population[person] == 3:
                    population[person] = 4

                sick_list_time[person, 0] = day
                sick_list_time[person, 1] = np.random.exponential(scale=1/gamma)

        # Recovery
        infected = all_indices[(population == 1) | (population == 4)]
        for person in infected:
            infection_day, duration = sick_list_time[person]
            if day > infection_day + duration:
                if population[person] == 1:
                    population[person] = 2
                    immune_list_time[person, 0] = day
                    immune_list_time[person, 1] = np.random.exponential(scale=immune_time)
                elif population[person] == 4:
                    population[person] = 3

        # Immunity waning (2 → 0)
        immune = all_indices[population == 2]
        waned = immune[day > (immune_list_time[immune, 0] + immune_list_time[immune, 1])]
        population[waned] = 0

        # Track population proportions
        susceptible_list.append(np.mean((population == 0) | (population == 3)))
        infected_list.append(np.mean(population == 1))
        infected_vaccine_list.append(np.mean(population == 4))
        vaccine_list.append(np.mean((population == 3) | (population == 4)))
        immune_list.append(np.mean(population == 2))

    return susceptible_list, infected_list, infected_vaccine_list, vaccine_list, immune_list


def plot_sim(vaccine_effectiveness: float = 0.9, save: bool = False):
    susceptible, infected, infected_vac, vaccinated, immune = run_simulation_with_vaccine(
        vaccine_effectiveness=vaccine_effectiveness
    )

    plt.figure(figsize=(10, 6))
    plt.title(f"Vaccine effectiveness={vaccine_effectiveness}")
    plt.plot(susceptible, color="blue", label="Susceptible")
    plt.plot(infected, color="red", label="Infected")
    plt.plot(vaccinated, color="magenta", label="Vaccine")
    plt.plot(infected_vac, color="green", label="Infected vaccine")
    plt.plot(immune, color="black", label="Immune")
    plt.grid(visible=True)
    plt.legend()

    if save:
        plt.savefig(f"vaccine_model_effectiveness_{vaccine_effectiveness:.2f}.png")
    else:
        plt.show()


def run_ci(vaccine_effectiveness: float = 0.9, num_runs: int = 10):
    max_infected_list = []

    for _ in range(num_runs):
        _, infected, infected_vac, _, _ = run_simulation_with_vaccine(
            vaccine_effectiveness=vaccine_effectiveness
        )
        max_infected = max(infected + infected_vac)
        max_infected_list.append(max_infected)

    data = np.array(max_infected_list)
    mean = np.mean(data)
    sem = stats.sem(data)
    ci = 0.95
    h = sem * stats.t.ppf((1 + ci) / 2., len(data) - 1)

    print(f"Peak mean infected people at one time: {mean:.4f}")
    print(f"95% CI: ({mean - h:.4f}, {mean + h:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vaccine-enhanced SIR Model Simulation")
    parser.add_argument("--ci", action="store_true", help="Run confidence interval computation")
    parser.add_argument("--plot", action="store_true", help="Plot the simulation")
    parser.add_argument("--save", action="store_true", help="Save plot instead of showing")
    parser.add_argument("--ve", type=float, default=0.5, help="Vaccine effectiveness (0-1)")

    args = parser.parse_args()

    if args.ci:
        run_ci(vaccine_effectiveness=args.ve)
    if args.plot:
        plot_sim(vaccine_effectiveness=args.ve, save=args.save)
