#补充实验
#<k>的影响W%
#10次循环
#ER网络
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def simulate_spread(G, infection_rate, base_c2_rate, recovery_rate, initial_infected=10, T=100):
    N = G.number_of_nodes()
    status = np.zeros(N)  # 0: Susceptible, 1: Infected
    initial_infected_nodes = np.random.choice(list(G.nodes()), initial_infected, replace=False)
    status[initial_infected_nodes] = 1

    for t in range(T):
        new_infected = []
        new_recovered = []
        for node in G.nodes():
            if status[node] == 0:
                neighbors = list(G.neighbors(node))
                infected_neighbors = sum(status[neighbor] for neighbor in neighbors)

                infection_prob = 1 - np.power((1 - infection_rate), infected_neighbors)
                if np.random.rand() < infection_prob:
                    new_infected.append(node)

                if infected_neighbors > 0:
                    c2_rate = base_c2_rate * infected_neighbors / len(neighbors)
                    if np.random.rand() < c2_rate:
                        new_infected.append(node)
            elif status[node] == 1:
                if np.random.rand() < recovery_rate:
                    new_recovered.append(node)

        # Update statuses using sets to ensure no duplicate changes
        new_infected_set = set(new_infected)
        status[list(new_infected_set)] = 1
        new_recovered_set = set(new_recovered)
        status[list(new_recovered_set)] = 0

    return np.sum(status == 1) / N  # Return the final infected ratio

# Parameters
infection_rate = 0.1
base_c2_rate = 0.8
recovery_rate = 1  # 100% chance of recovery per day
T = 100
N = 1000
initial_infected = 10
simulations = 20  # Number of simulations to average

k_values = range(2, 70, 6)  # Adjusted to give a range of <k> values.

# Collect results
avg_final_infection_ratios = []
for k in k_values:
    infection_ratios = []
    for _ in range(simulations):
        # Calculate the probability p for the ER network based on the desired average degree <k>
        p = k / (N - 1)
        G = nx.erdos_renyi_graph(N, p)  # Generate an ER network with N nodes and probability p
        w = simulate_spread(G, infection_rate, base_c2_rate, recovery_rate, initial_infected, T)
        infection_ratios.append(w)
    avg_w = np.mean(infection_ratios)
    avg_final_infection_ratios.append(avg_w)

# Plotting
plt.figure(figsize=(8, 8))
plt.plot(k_values, avg_final_infection_ratios, marker='o', color='blue', linewidth=2)
plt.xlabel('<k>',fontsize=22)
plt.ylabel(' W%',fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlim(0,70)
plt.ylim(-0.01,0.6)
plt.show()
