from time import sleep
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from copy import copy
from itertools import islice

def link_cost(G,x):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return list(map(lambda u, v, d, x: d['a'] + d['b']*(x/d['c'])**d['n'], u,v,d,x))

def beckmann(G, x):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return sum(list(map(lambda u, v, d, x: x*(d['a'] + (d['b']*(x/d['c'])**d['n'])/(d['n']+1)), u,v,d,x)))
  
def total(G,x):
    [u,v,d] = [list(t) for t in zip(*list(sorted(G.edges(data=True))))]
    return sum(list(map(lambda u, v, d, x: x*(d['a'] + (d['b']*(x/d['c'])**d['n'])), u,v,d,x)))

def label_edges_with_id(G):
    for index, (u,v) in enumerate(sorted(G.edges(), key= lambda edge: (edge[0], edge[1]))):
        G[u][v]['id'] = index

def get_np_array_from_edge_attribute(G, name):
    return np.array([value for (key, value) in sorted(nx.get_edge_attributes(G, name).items())])

def update_edge_attribute(G, name, vector):
    d = dict(zip(sorted(G.edges()), vector))
    nx.set_edge_attributes(G, d, name)

def _all_or_nothing(G, od, paths):
    y = np.zeros(len(G.edges())) 
    od = od.tocoo()
    # for demand (d) for s and t
    for s,t,d in (zip(od.row, od.col, od.data)):
        #print (s,t,d)
        path = nx.shortest_path(G, s, t, weight='weight')
        #print(path)
        if (s,t) in paths.keys():
            if not path in paths[(s,t)]:
                paths[(s,t)].append(path)
        else:
            paths[(s,t)] = [path]
        #print(path)
        for u,v in zip(path[:-1], path[1:]):
            edge_id = G[u][v]['id']
            #print(edge_id)
            y[edge_id] += d
    return (y, paths)

def _line_search(G, x, y, func, ls_e):
    p = 0
    q = 1
    while True:
        alpha = (p+q)/2.0
        D_alpha = sum((y-x)*link_cost(G, x + alpha*(y-x)))
        #print( D_alpha)
        if D_alpha <= 0:
            p = alpha
        else:
            q = alpha

        if q-p < ls_e:
            break

    return x + alpha*(y-x)

def weighted_path_cost(G, path):
    return sum([G[u][v]['weight'] for (u,v) in zip(path[:-1], path[1:])])

def fw(G, od, max_iter=10, conv_e=0.01, ls_e=0.05, func=link_cost):
    lbd = 0
    label_edges_with_id(G)
    x = np.zeros(len(G.edges()))
    update_edge_attribute(G, 'x', x)
    update_edge_attribute(G, 'weight', func(G,x))
    
    # step 1 - perform first all or nothing assignment
    (x, paths) = _all_or_nothing(G, od, {})


    update_edge_attribute(G, 'x', x)
    update_edge_attribute(G, 'weight', func(G,x))

    for i in range(max_iter):
        if i % 1000 == 0:
            for (u,v), value in paths.items():
                print('getting costs of paths between {} and {}'.format(u,v))
                for path in value:
                    print(path)
                    print(weighted_path_cost(G, path))
                    break
        # step 2 - perform next all or nothing assignment
        (y, paths) = _all_or_nothing(G, od, paths)

        next_x = _line_search(G, x, y, func, ls_e)

        
        update_edge_attribute(G, 'x', x)
        update_edge_attribute(G, 'weight', func(G,x))

        x = next_x

    return G, x, paths

''' helper function for getting k paths '''

def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))