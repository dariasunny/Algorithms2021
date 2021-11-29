import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import sys
import networkx as nx
import scipy
import pandas as pd
import scipy.sparse
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

#number 1
def create_spanning_tree(graph, starting_vertex):
    mst = defaultdict(set)
    visited = set([starting_vertex])
    edges = [
        (cost, starting_vertex, to)
        for to, cost in graph[starting_vertex].items()
    ]
    heapq.heapify(edges)
    heap_size = sys.getsizeof(edges)
    while edges:
        cost, frm, to = heapq.heappop(edges)
        if to not in visited:
            visited.add(to)
            mst[frm].add(to)
            for to_next, cost in graph[to].items():
                if to_next not in visited:
                    heapq.heappush(edges, (cost, to, to_next))
    space = sys.getsizeof(mst) + heap_size + sys.getsizeof(graph)
    return mst, space
  
def transform_row(small_dict):
  new_dict = {}
  for k in small_dict:
    new_dict[k] = small_dict[k]['weight']
  return new_dict

def transform_graph(big_dict):
  new_dict_for_big = {}
  for k in big_dict:
    new_dict_for_big[k] = transform_row(big_dict[k])
  return new_dict_for_big

def add_random_weights(graph):
  for i in graph.nodes:
    for j in graph[i]:
      graph[i][j]['weight'] = random.randint(1,10)
      
def nlogn(v):
  return (v**2)*math.log(v)*0.000313/15201.8

def kvadrat(v):
  return v**2 * 0.7
average = []
average_space = []
reps = 100
n_range = list(range(reps))
problem_sizes = range(10,500,10)
for m in problem_sizes:
  time_raw = []
  space_raw = []
  for n in n_range:
      x = abs(np.random.randn(1,n))
      graph = nx.generators.random_graphs.erdos_renyi_graph(m, 0.1)
      GRAPH_SIZE = sys.getsizeof([e for e in graph.edges]) + sys.getsizeof([e for e in graph.nodes])
      add_random_weights(graph)
      tart_time = time.perf_counter()
      # create random graph of size m.
      t, s = create_spanning_tree(transform_graph(graph), 1)
      time_raw.append((time.perf_counter() - tart_time))
      space_raw.append(s+GRAPH_SIZE)
  average.append(sum(time_raw)/reps)
  #print(space_raw)
  average_space.append(sum(space_raw)/reps)
#print(average_space)
plt.plot(problem_sizes, average, label='experimental')
plt.plot(problem_sizes, [nlogn(x) for x in problem_sizes], label='theoretical (e*log(v))')
#plt.plot(problem_sizes, [kvadrat(x) for x in problem_sizes], label='theoretical(n^2)')
# naming the x axis
plt.xlabel('Number of vertices')
# naming the y axis
plt.ylabel('Average time')
# giving a title to my graph
plt.title('Time complexity')
plt.legend()
plt.show()

plt.plot(problem_sizes, average_space, label='experimental')
plt.plot(problem_sizes, [kvadrat(x) for x in problem_sizes], label='theoretical')
# naming the x axis
plt.xlabel('Number of vertices')
# naming the y axis
plt.ylabel('Average space')
# giving a title to my graph
plt.title('Space complexity')
plt.legend()
plt.show()

#small example for fan 
number_of_vertex = 9
number_of_edges = 16

adj_matrix = np.zeros((number_of_vertex ,number_of_vertex ))
for n in range(number_of_edges):
  i = np.random.randint(number_of_vertex)
  j = np.random.randint(number_of_vertex )
  while i == j or adj_matrix[i][j] > 0:
    i = np.random.randint(number_of_vertex)
    j = np.random.randint(number_of_vertex)
  adj_matrix[i][j] = np.random.randint(100) +1
  adj_matrix[j][i] = adj_matrix[i][j]
  
adj_sparse = scipy.sparse.coo_matrix(adj_matrix, dtype=np.int8)
labels = range(0,number_of_vertex)
DF_adj = pd.DataFrame(adj_sparse.toarray(),index=labels,columns=labels)
print(DF_adj)

#just check that ok
n = 0
for i in range(number_of_vertex):
  for j in range(number_of_vertex):
    if DF_adj[i][j] > 0:
      n += 1
print(n)

H=nx.Graph(adj_matrix) 
pos=nx.spring_layout(H)
nx.draw_networkx(H,pos)
labels = nx.get_edge_attributes(H,'weight')
nx.draw_networkx_edge_labels(H,pos,edge_labels=labels)


H=nx.Graph(adj_matrix) 
pos=nx.spring_layout(H)
nx.draw_networkx(H,pos)
labels = nx.get_edge_attributes(H,'weight')
nx.draw_networkx_edge_labels(H,pos,edge_labels=labels)
plt.title("Original graph")

slovar = transform_graph(H.adj)
som = create_spanning_tree(slovar, 1)
pos=nx.spring_layout(H)
color_map = []
for edge in H.edges:
    if edge in [(1, 7), (4, 7), (0, 4), (0, 6), (3, 6), (5, 6), (2, 5), (5, 8)]:
        color_map.append('red')
    else: 
        color_map.append('grey') 

nx.draw(H, pos = pos,
        edge_color=color_map,
        with_labels=True)
plt.title("Minimal spanning tree")


#number 2

def bfs(graph, source, sink):
    queue, visited = [(source, [source])], [source]
    while queue:
        u, path = queue.pop(0)
        edge_nodes = set(graph[u].keys()) - set(path)
        for v in edge_nodes:
            if v in visited:
                continue
            visited.append(v)
            if not graph.has_edge(u, v):
                continue
            elif v == sink:
                return path + [v]
            else:
                queue.append((v, path + [v]))
                
def augment_flow(graph, flow_path):
    bottleneck = min([graph[u][v]['weight'] for u, v in flow_path])
    for u, v in flow_path:
        updated_capacity = graph[u][v]['weight'] - bottleneck
        #print(updated_capacity)
        if updated_capacity:
            graph[u][v]['weight'] = updated_capacity
        else:
            #print("removing " + str(u) + str(v))
            graph.remove_edge(u, v)
        if not graph.has_edge(v, u):
            graph.add_edge(v, u)
            graph[v][u]['weight'] = 0
        graph[v][u]['weight'] += bottleneck
    return graph
  
def ford_fulkerson(graph, source, sink):
    path = bfs(graph, source, sink)
    while path:
        #print(path)
        flow_path = list(zip(path[:-1], path[1:]))
        graph = augment_flow(graph, flow_path)
        path = bfs(graph, source, sink)
    
    return graph, sys.getsizeof([node for node in graph.nodes]) + sys.getsizeof([node for node in graph.edges]) 
  
def calc_flow(G_f, G):
    flow = {i: {} for i in G}
    for u, v in G.edges:
        if G_f.has_edge(u, v):
            f = G[u][v]['weight'] - G_f[u][v]['weight']
            flow[u][v] = max(0, f)
        else:
            flow[u][v] = G[u][v]['weight']
    return flow
  
def inv_transform_row(small_dict):
  new_dict = {}
  for k in small_dict:
    if small_dict[k] !=0:
      new_dict[k] = {'weight' : small_dict[k]}
    else:
      new_dict[k] = {}
  return new_dict

def inv_transform_graph(big_dict):
  new_dict_for_big = {}
  for k in big_dict:
    new_dict_for_big[k] = inv_transform_row(big_dict[k])
  return new_dict_for_big

def nlogn(v):
  return v**3*0.0002213/15201.8
#*math.log(v)
def kvadrat(v):
  return v**2 * 0.8
average = []
average_space = []
reps = 100
n_range = list(range(reps))
problem_sizes = range(10,200,10)
for m in problem_sizes:
  time_raw = []
  space_raw = []
  for n in n_range:
      x = abs(np.random.randn(1,n))
      adj_matrix = np.random.randint(low = 1, high=10, size=(m,m), dtype=int)
      graph = nx.generators.random_graphs.erdos_renyi_graph(m, 0.1)
      add_random_weights(graph)
      DG1=nx.DiGraph(graph) 
      DG2=nx.DiGraph(graph)
      tart_time = time.perf_counter()
      # create random graph of size m.
      g, s = ford_fulkerson(DG1, 1, 5)
      DG3 = calc_flow(DG1, DG2)
      time_raw.append((time.perf_counter() - tart_time))
      space_raw.append(s)
  average.append(sum(time_raw)/reps)
  #print(space_raw)
  average_space.append(sum(space_raw)/reps)
#print(average_space)
plt.plot(problem_sizes, average, label='experimental')
plt.plot(problem_sizes, [nlogn(x) for x in problem_sizes], label='theoretical')
#plt.plot(problem_sizes, [kvadrat(x) for x in problem_sizes], label='theoretical(n^2)')
# naming the x axis
plt.xlabel('Number of vertices')
# naming the y axis
plt.ylabel('Average time')
# giving a title to my graph
plt.title('Time complexity')
plt.legend()
plt.show()

plt.plot(problem_sizes, average_space, label='experimental')
plt.plot(problem_sizes, [kvadrat(x) for x in problem_sizes], label='theoretical')
# naming the x axis
plt.xlabel('Number of vertices')
# naming the y axis
plt.ylabel('Average space')
# giving a title to my graph
plt.title('Space complexity')
plt.legend()
plt.show()

#small example

number_of_vertex = 9
number_of_edges = 16

adj_matrix = np.zeros((number_of_vertex ,number_of_vertex ))
for n in range(number_of_edges):
  i = np.random.randint(number_of_vertex)
  j = np.random.randint(number_of_vertex )
  while i == j or adj_matrix[i][j] > 0:
    i = np.random.randint(number_of_vertex)
    j = np.random.randint(number_of_vertex)
  adj_matrix[i][j] = np.random.randint(100) +1
  adj_matrix[j][i] = adj_matrix[i][j]
  
adj_sparse = scipy.sparse.coo_matrix(adj_matrix, dtype=np.int8)
labels = range(0,number_of_vertex)
DF_adj = pd.DataFrame(adj_sparse.toarray(),index=labels,columns=labels)
print(DF_adj)

DG = nx.DiGraph(adj_matrix) 
pos=nx.spring_layout(DG)
nx.draw_networkx(DG,pos)
labels = nx.get_edge_attributes(DG,'weight')
nx.draw_networkx_edge_labels(DG,pos,edge_labels=labels)

DG1=nx.DiGraph(adj_matrix) 
DG2=nx.DiGraph(adj_matrix)
ford_fulkerson(DG1, 0, 4)
DG3 = calc_flow(DG1, DG2)

DG4 = inv_transform_graph(DG3)
DG4 = nx.DiGraph(DG4)


pos=nx.spring_layout(DG4)
nx.draw_networkx(DG4,pos)
labels = nx.get_edge_attributes(DG4,'weight')
nx.draw_networkx_edge_labels(DG4,pos,edge_labels=labels)
