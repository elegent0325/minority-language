#fig6a
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


#先感染再自发
def simulate_spread(G, infection_rate, base_c2_rate, initial_infected=10, T=100):
    N = G.number_of_nodes()
    status = np.zeros(N)  # 0: Susceptible, 1: Infected
    initial_infected_nodes = np.random.choice(G.nodes(), initial_infected, replace=False)
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
                
            elif status[node] == 1 and np.random.rand() < 1:
                new_recovered.append(node)
        
        status[new_infected] = 1
        status[new_recovered] = 0

    return np.sum(status == 1) / N


infection_rate_values = np.linspace(0,0.1,20)
base_c2_rate_values = np.linspace(0, 1,20)
c1, c2 = np.meshgrid(infection_rate_values, base_c2_rate_values)
N = 1000
m=3
G = nx.barabasi_albert_graph(N, m)
num_simulations = 20

S_counts = np.zeros_like(c1)
for simulation in range(num_simulations):
    for i in range(len(infection_rate_values)):
        for j in range(len(base_c2_rate_values)):
            S_counts [i, j] += simulate_spread(G, c1[i, j], c2[i, j])
            
S_averages = S_counts  / num_simulations



fig = plt.figure(figsize=(15, 15))#图的大小
ax = fig.add_subplot(111, projection='3d')#绘制三维图
surf = ax.plot_surface(c1,c2, S_averages, cmap='rainbow',edgecolor='none')#具体

cbar = fig.colorbar(surf, shrink=0.5, aspect=15, pad=0.14)#显色条
cbar.ax.tick_params(labelsize=24)  # Adjust color bar tick label size


ax.set_xlabel(' C', fontsize=24, labelpad=24)
ax.set_ylabel(' F', fontsize=24, labelpad=24)
ax.set_zlabel('w% ', fontsize=24, labelpad=24)

ax.tick_params(axis='x', labelsize=24, pad=10)
ax.tick_params(axis='y', labelsize=24, pad=10)
ax.tick_params(axis='z', labelsize=24, pad=10)
ax.set_zlim(0, 0.6)  # Sets the limits for the z-axis
ax.view_init(elev=30, azim=120)  # Sets the elevation and azimuthal angles
ax.dist = 9
plt.show()
