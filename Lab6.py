import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import scipy.sparse
import time

#number 1
number_of_vertex = 100
number_of_edges = 500

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

#calculate dij alg average time
time_raw = []
start = np.random.randint(number_of_vertex)
for a in range(1,11):
  start_time = time.perf_counter()
  nx.single_source_dijkstra_path(H, start, cutoff=None, weight='weight')
  time_raw.append((time.perf_counter() - start_time))
average = sum(time_raw)/10
average

#calculate bf time
time_raw = []
start = np.random.randint(number_of_vertex)
for a in range(1,11):
  start_time = time.perf_counter()
  nx.single_source_bellman_ford_path(H, np.random.randint(10), weight='weight')
  time_raw.append((time.perf_counter() - start_time))
average = sum(time_raw)/10
average

#number 2
import networkx as nx
from matplotlib import pyplot as plt

G = nx.grid_2d_graph(10,20)

plt.figure(figsize=(10,10))
pos = {(x,y):(y,-x) for x,y in G.nodes()}
nx.draw(G, pos=pos, 
        node_color='grey', 
        with_labels=False,
        node_size=60)

#make some removed cells
for n in range(40):
  i = np.random.randint(10)
  j = np.random.randint(20)
  while (i,j) not in list(G.nodes):
    i = np.random.randint(10)
    j = np.random.randint(20)
  G.remove_node((i,j))
  
#graph 
nx.draw(G, pos=pos, 
      node_color='grey', 
      with_labels=False,
      node_size=60)

#Manchattan heuristic
time_raw = []
for a in range(1,6):
  start = (np.random.randint(10), np.random.randint(20))
  end = (np.random.randint(10), np.random.randint(20))
  while start not in list(G.nodes):
    start = (np.random.randint(10), np.random.randint(20))
  while end not in list(G.nodes):
    end = (np.random.randint(10), np.random.randint(20))
  start_time = time.perf_counter()
  nx.astar_path(G, start, end, heuristic = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]))
  #print(start)
  #print(end)
  #print(nx.astar_path(G, start, end, heuristic = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1])))
  #print(nx.astar_path(G, start, end, heuristic = lambda x, y: math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)))
  time_raw.append((time.perf_counter() - start_time))
average = sum(time_raw)/5
average

#EU heuristic
time_raw = []
for a in range(1,6):
  start = (np.random.randint(10), np.random.randint(20))
  end = (np.random.randint(10), np.random.randint(20))
  while start not in list(G.nodes):
    start = (np.random.randint(10), np.random.randint(20))
  while end not in list(G.nodes):
    end = (np.random.randint(10), np.random.randint(20))
  start_time = time.perf_counter()
  nx.astar_path(G, start, end, heuristic = lambda x, y: math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2))
  #print(start)
  #print(end)
  #print(nx.astar_path(G, start, end, heuristic = lambda x, y: math.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)))
  time_raw.append((time.perf_counter() - start_time))
average = sum(time_raw)/5
average


color_map = []
for node in G:
    if node in [(0, 18), (1, 18), (2, 18), (3, 18), (4, 18), (4, 17), (4, 16), (4, 15), (4, 14), (4, 13), (4, 12), (4, 11), (4, 10), (4, 9), (4, 8), (4, 7), (4, 6), (4, 5), (5, 5), (5, 4)]:
        color_map.append('red')
    else: 
        color_map.append('grey') 

nx.draw(G, pos=pos, 
        node_color=color_map, 
        with_labels=False,
        node_size=60)
