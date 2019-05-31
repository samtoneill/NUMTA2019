import networkx as nx
import matplotlib.pyplot as plt
from agents import *
from fw_tap import *
from bandits import *

G = nx.grid_2d_graph(4,4,periodic=False) 
G = nx.DiGraph(G)
G = nx.convert_node_labels_to_integers(G)
pos = nx.spring_layout(G, iterations=100)

label_edges_with_id(G)

GBanditGame = GraphBPRBanditGame(G, T =10)
GBanditGame.seed(seed=2)
GBanditGame.reset()

ori = [0,2,3,7]
des = [15,13,12,8]
dem = [75,50,75,50]


results_egreedysemi = []
results_egreedyfull = []
results_ewsemi = []
results_ewfull = []
min_od_costs = []

for seed in range(1,2):
  print('Seed = {}'.format(seed))
  GBanditGame.seed(seed=seed)
  GBanditGame.reset()

  od = csr_matrix((dem, (ori, des)))

  G_FW, x, paths = fw(GBanditGame.G, od, max_iter=400)

  
  min_od_cost = []
  for (u,v), value in paths.items():
      path_costs = []
      for path in value:
          path_costs.append(weighted_path_cost(GBanditGame.G, path))
      min_od_cost.append(np.min(path_costs))

  equi_beckmann = beckmann(GBanditGame.G,x)
  equi_total = total(GBanditGame.G,x)
  min_od_costs.append(min_od_cost)

  
  
  k = 10
  alpha = 4
  epsilon = .1
    
  for i in range(2):
    print('Iteration = {}'.format(i))
    
    # Reset game with given seed and run episode for e-greedy semi-bandit
    GBanditGame.seed(seed=seed)
    GBanditGame.reset()
    ods = [[EGreedySemiAgent(GBanditGame, k_shortest_paths(GBanditGame.G, o, d, k, weight='weight'), epsilon) for _ in range(demand)] for (o,d,demand) in zip(ori, des, dem)]
    od_rewards, edge_info = algorithm_loop(GBanditGame, ods)
    results_egreedysemi.append((od_rewards, edge_info, min_od_cost, equi_total, equi_beckmann))
    
    # Reset game with given seed and run episode for e-greedy bandit
    GBanditGame.seed(seed=seed)
    GBanditGame.reset()
    ods = [[EGreedyFullAgent(GBanditGame, k_shortest_paths(GBanditGame.G, o, d, k, weight='weight'), epsilon) for _ in range(demand)] for (o,d,demand) in zip(ori, des, dem)]
    od_rewards, edge_info = algorithm_loop(GBanditGame, ods)
    results_egreedyfull.append((od_rewards, edge_info, min_od_cost, equi_total, equi_beckmann))

    # Reset game with given seed and run episode for ew semi-bandit
    GBanditGame.seed(seed=seed)  
    GBanditGame.reset()
    gammas = [1*(1/i**(1/alpha)) for i in range(1,GBanditGame.T+1)]
    ods = [[EWSemiAgent(GBanditGame, k_shortest_paths(GBanditGame.G, o, d, k, weight='weight')) for _ in range(demand)] for (o,d,demand) in zip(ori, des, dem)]
    od_rewards, edge_info = algorithm_loop(GBanditGame, ods, gammas=gammas)
    results_ewsemi.append((od_rewards, edge_info, min_od_cost, equi_total, equi_beckmann))

    # Reset game with given seed and run episode for ew bandit
    GBanditGame.seed(seed=seed)
    GBanditGame.reset()
    ods = [[EWFullAgent(GBanditGame, k_shortest_paths(GBanditGame.G, o, d, k, weight='weight')) for _ in range(demand)] for (o,d,demand) in zip(ori, des, dem)]
    od_rewards, edge_info = algorithm_loop(GBanditGame, ods, gammas=gammas)
    results_ewfull.append((od_rewards, edge_info, min_od_cost, equi_total, equi_beckmann))

plt.semilogy(np.sum(np.mean(np.array([od_rewards for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_ewsemi]),axis=0), axis=1), label='EW-SB')
plt.semilogy(np.sum(np.mean(np.array([od_rewards for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_ewfull]),axis=0), axis=1), label='EW-B')
plt.semilogy(np.sum(np.mean(np.array([od_rewards for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_egreedysemi]),axis=0), axis=1), label=r'$\epsilon$G-SB')
plt.semilogy(np.sum(np.mean(np.array([od_rewards for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_egreedyfull]),axis=0), axis=1), label=r'$\epsilon$G-B')
plt.semilogy([0,GBanditGame.T], [np.mean(np.array([equi_total for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_ewsemi])), np.mean(np.array([equi_total for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_ewsemi]))], label=r'Potential function $\Phi$')
plt.xlabel('Round (t)', fontsize=18)
plt.ylabel('Total Cost', fontsize=18)
plt.legend()
plt.savefig('logtotal.eps', format='eps')
plt.savefig('logtotal.png', format='png')

fig, ax = plt.subplots()
plt.bar(range(np.sum(dem)), np.mean(np.array([od_rewards for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_ewsemi]),axis=0)[0])
for index, (a,b) in enumerate(zip(np.insert(np.cumsum(dem),0, 0), np.insert(np.cumsum(dem),0, 0)[1:])):
  plt.plot([a,b], [np.mean(min_od_costs,axis=0)[index], np.mean(min_od_costs,axis=0)[index]], color='red')
ax.set_yscale('log')
plt.ylabel('Player Cost', fontsize=18)
plt.xlabel('Player $i$', fontsize=18)

ylim=plt.ylim()

plt.savefig('equi_0.eps', format='eps')
plt.savefig('equi_0.png', format='png')

fig, ax = plt.subplots()
plt.bar(range(np.sum(dem)), np.mean(np.array([od_rewards for od_rewards, edge_info, min_od_cost,equi_total, equi_beckmann in results_ewsemi]),axis=0)[-1])
for index, (a,b) in enumerate(zip(np.insert(np.cumsum(dem),0, 0), np.insert(np.cumsum(dem),0, 0)[1:])):
  plt.plot([a,b], [np.mean(min_od_costs,axis=0)[index], np.mean(min_od_costs,axis=0)[index]], color='red')
ax.set_yscale('log')
plt.ylabel('Player Cost', fontsize=18)
plt.xlabel('Player $i$', fontsize=18)
  
plt.ylim(ylim)
  
plt.savefig('equi_T.eps', format='eps')
plt.savefig('equi_T.png', format='png')