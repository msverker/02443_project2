import numpy as np
import heapq
import matplotlib.pyplot as plt
from tqdm import tqdm

class CovidSimulation:
    def __init__(self, beta, gamma, population, death_ratio):
        self.n = population
        self.mean_sick_time = 1/gamma
        self.mean_infection_probability = beta
        self.init_sick_num = 100
        self.hospital_capacity_per_day = 1e3
        self.death_dist = [death_ratio, 1 - death_ratio]

    def generate_sick_time(self):
        return np.random.exponential(scale=self.mean_sick_time)
    
    def generate_interarrival_time(self, num_sick, num_healthy):
        infection_rate = self.mean_infection_probability * num_sick * (num_healthy - num_sick) / num_healthy
        if infection_rate <= 0:
            return float('inf')
        return np.random.exponential(1 / infection_rate)
        
    def simulate(self):
        num_sick = self.init_sick_num
        sick_info = []
        recovered_info = []
        death_info = []
        healthy_info = []
        num_recovered = 0
        num_dead = 0
        num_healthy = self.n - num_sick - num_recovered - num_dead
        current_time = 0.0
        max_time = 200
        event_queue = []

        next_arrival = current_time + self.generate_interarrival_time(num_sick, num_healthy)
        heapq.heappush(event_queue, (next_arrival, 'arrival'))

        while current_time <= max_time:
            event_time, event_type = heapq.heappop(event_queue)
            current_time = event_time

            if event_type == 'arrival':
                if num_healthy > 0:
                  num_sick += 1
                  sick_info.append([current_time, num_sick])
                  num_healthy -= 1
                  healthy_info.append([current_time, num_healthy])
                  service_time = self.generate_sick_time()
                  departure_time = current_time + service_time
                  heapq.heappush(event_queue, (departure_time, 'departure'))

                  next_arrival = current_time + self.generate_interarrival_time(num_sick, num_healthy)
                  heapq.heappush(event_queue, (next_arrival, 'arrival'))
                else:
                    print('EVERYONE IS SICK')
                    break

            elif event_type == 'departure' and num_sick > 0:
                num_sick -= 1
                sick_info.append([current_time, num_sick])
                prob = np.random.choice([0,1], p=self.death_dist)
                if prob == 1:
                    num_recovered += 1
                else:    
                    num_dead +=1
                recovered_info.append([current_time, num_recovered])
                death_info.append([current_time, num_dead])

        return num_sick, num_recovered, sick_info, recovered_info, healthy_info, death_info


if __name__ == '__main__':
    total_pop = 180000
    death_ratio = 0.04
    hospital_capacity = 7500
    hospitalized_ratio = 0.20

    exceeding_cases = []    
    non_exceeding_cases = []

    plt.figure(figsize=(12, 8))
    i = 0

    with tqdm(total=5, desc="Collecting cases") as pbar:
        while len(exceeding_cases) < 5:
            beta = np.random.uniform(0, 1)
            gamma = np.random.uniform(0, 1)

            covid_sim = CovidSimulation(beta, gamma, total_pop, death_ratio)
            _, _, sick_info, _, _, _ = covid_sim.simulate()

            sick_times = [t for t, _ in sick_info]
            sick_vals = [hospitalized_ratio * s for _, s in sick_info]
            max_hospitalized = max(sick_vals)
            
            if max_hospitalized > hospital_capacity and len(exceeding_cases) < 5:
                exceeding_cases.append((beta, gamma, sick_info))
                color = 'red'
                pbar.update(1)
            else:
                continue
        

            plt.plot(sick_times, sick_vals, color=color, alpha=0.4)

            i += 1
        covid_sim_default = CovidSimulation(0.1313, 0.07, total_pop, death_ratio)
        _, _, sick_info, _, _, _ = covid_sim_default.simulate()
        sick_times_d = [t for t, _ in sick_info]
        sick_vals_d = [hospitalized_ratio * s for _, s in sick_info]

        plt.plot(sick_times_d, sick_vals_d, color = 'green', alpha = 0.4)

    # Add hospital capacity line
    plt.axhline(y=hospital_capacity, color='black', linestyle='--', linewidth=2, label='Hospital Capacity')

    # Annotate up to 2 exceeding cases
    for idx, (beta, gamma, new_sick_info) in enumerate(exceeding_cases[:2]):
        max_day, max_sick = max(new_sick_info, key=lambda x: x[1])
        max_hospitalized = hospitalized_ratio * max_sick
        plt.text(max_day, max_hospitalized + 500, f'β={beta:.2f}, γ={gamma:.2f}', fontsize=9, color='red')

    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Sick Individuals')
    plt.title('COVID-19 Active Infections vs Hospital Capacity')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/sick_curves_vs_capacity2.png')
    plt.show()

    print(f"\nCollected {len(exceeding_cases)} exceeding and {len(non_exceeding_cases)} non-exceeding scenarios (total simulations run: {i})")

    print("\nExample exceeding cases (β, γ):")
    for beta, gamma, _ in exceeding_cases:
        min_ratio = min(beta / gamma for beta, gamma, _ in exceeding_cases)
    print(f"  min(β/γ) = {min_ratio:.4f}")

