import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class SIRGraph:
    def __init__(self, num_nodes, infection_probs=None, recovery_prob=0.05):
        """
        infection_probs: dict like {'family': 0.2, 'work': 0.05}
        """
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_nodes))
        self.recovery_prob = recovery_prob
        self.infection_probs = infection_probs or {'family': 0.2, 'work': 0.05}
        self.states = {node: 'S' for node in self.graph.nodes}

    def add_edge(self, u, v, edge_type='family'):
        """Add edge with a type (e.g., 'family' or 'work')"""
        self.graph.add_edge(u, v, type=edge_type)

    def infect_node(self, node):
        if self.states[node] == 'S':
            self.states[node] = 'I'

    def step(self):
        new_states = self.states.copy()
        
        for node in self.graph.nodes:
            if self.states[node] == 'I':
                for neighbor in self.graph.neighbors(node):
                    if self.states[neighbor] == 'S':
                        edge_type = self.graph.edges[node, neighbor].get('type', 'family')
                        infection_prob = self.infection_probs.get(edge_type, 0.1)
                        if random.random() < infection_prob:
                            new_states[neighbor] = 'I'

                if random.random() < self.recovery_prob:
                    new_states[node] = 'R'

        self.states = new_states

    def get_counts(self):
        counts = {'S': 0, 'I': 0, 'R': 0}
        for state in self.states.values():
            counts[state] += 1
        return counts

    def run(self, steps):
        history = []
        for _ in range(steps):
            self.step()
            history.append(self.get_counts())
        return history

class SIRGraphAnimated(SIRGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def animate(self, steps=50, interval=500):
      """Animate the simulation + live SIR curve"""
      pos = nx.spring_layout(self.graph, seed=42)
      fig, (ax_graph, ax_plot) = plt.subplots(1, 2, figsize=(14, 6))

      history = []

      line_s, = ax_plot.plot([], [], label='Susceptible', color='skyblue')
      line_i, = ax_plot.plot([], [], label='Infected', color='red')
      line_r, = ax_plot.plot([], [], label='Recovered', color='green')
      ax_plot.set_xlim(0, steps)
      ax_plot.set_ylim(0, len(self.graph.nodes))
      ax_plot.set_xlabel("Days")
      ax_plot.set_ylabel("Number of People")
      ax_plot.set_title("SIR Model over Time")
      ax_plot.grid(True)
      ax_plot.legend()

      def update(frame):
          ax_graph.clear()
          self.step()
          history.append(self.get_counts())

          # Update network visualization
          colors = [self._get_color(node) for node in self.graph.nodes]
          nx.draw(self.graph, pos, node_color=colors, with_labels=False, ax=ax_graph,
                  edge_color='gray', node_size=100)
          ax_graph.set_title(f"Step {frame}")

          # Update time-series plot
          S = [day['S'] for day in history]
          I = [day['I'] for day in history]
          R = [day['R'] for day in history]
          line_s.set_data(range(len(S)), S)
          line_i.set_data(range(len(I)), I)
          line_r.set_data(range(len(R)), R)
          
          if frame in [0, steps // 2, steps - 1]:
            plt.savefig(f"sir_test_{frame:03d}.png")

          return line_s, line_i, line_r

      ani = animation.FuncAnimation(fig, update, frames=steps, interval=interval, repeat=False)
      plt.tight_layout()
      plt.show()


    def _get_color(self, node):
        return {'S': 'skyblue', 'I': 'red', 'R': 'green'}[self.states[node]]

if __name__ == '__main__':
    NUM_PEOPLE = 100
    FAMILY_SIZE = 4
    WORKGROUP_SIZE = 10
    public_restricted = True
    sir = SIRGraphAnimated(num_nodes=NUM_PEOPLE, infection_probs={'family': 0.1, 'work': 0.05, 'public_transport': 0.01}, recovery_prob=0.05)

    families = [list(range(i, i + FAMILY_SIZE)) for i in range(0, NUM_PEOPLE, FAMILY_SIZE)]

    for family in families:
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                sir.add_edge(family[i], family[j], edge_type='family')

    workers = [random.choice(family) for family in families]
    workgroups = [workers[i:i + WORKGROUP_SIZE] for i in range(0, len(workers), WORKGROUP_SIZE)]
    
    public_transport = [random.choice(family) for family in families]
    public_transport_minimized = np.random.choice(public_transport, size=5, replace=False)

    for group in workgroups:
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sir.add_edge(group[i], group[j], edge_type='work')

    if public_restricted:
        public_transport = public_transport_minimized
    for i in range(len(public_transport)):
      for j in range(i + 1, len(public_transport)):
          sir.add_edge(public_transport[i], public_transport[j], edge_type='transport')

    sir.infect_node(0)
    sir.animate(steps=100, interval=100)  # You can adjust timing here
    # history = sir.run(steps=100)
    # plot_sir_history(history)

