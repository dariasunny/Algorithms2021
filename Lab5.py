import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import scipy.sparse
from collections import defaultdict
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

adj_matrix = np.zeros((100,100))
for n in range(200):
  i = np.random.randint(100)
  j = np.random.randint(100)
  while i == j or adj_matrix[i][j] == 1:
    j = np.random.randint(100)
    i = np.random.randint(100)
  adj_matrix[i][j] = 1 
  adj_matrix[j][i] = 1
  
adj_sparse = scipy.sparse.coo_matrix(adj_matrix, dtype=np.int8)
labels = range(0,100)
DF_adj = pd.DataFrame(adj_sparse.toarray(),index=labels,columns=labels)
print(DF_adj)


#just check that it is ok
n = 0
for i in range(100):
  for j in range(100):
    if DF_adj[i][j]==1:
      n += 1
print(n)

#Network graph
G = nx.Graph()
G.add_nodes_from(labels)

#Connect nodes
for i in range(DF_adj.shape[0]):
    col_label = DF_adj.columns[i]
    for j in range(DF_adj.shape[1]):
        row_label = DF_adj.index[j]
        node = DF_adj.iloc[i,j]
        if node == 1:
            G.add_edge(col_label,row_label)


#Draw graph
nx.draw(G,with_labels = True)
plt.show() 

H=nx.Graph(adj_matrix) 

# converts from adjacency matrix to adjacency list
def convert(a):
    adjList = defaultdict(list)
    for i in range(len(a)):
        for j in range(len(a[i])):
                       if a[i][j]== 1:
                           adjList[i].append(j)
    return adjList
 
# driver code
AdjList = convert(adj_matrix)
print("Adjacency List:")
# print the adjacency list
for i in AdjList:
    print(i, end ="")
    for j in AdjList[i]:
        print(" -> {}".format(j), end ="")
    print()
    
#here df search 
visited = set() # Set to keep track of visited nodes of graph.

def dfs(visited, graph, node):  #function for dfs 
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)
            
# Driver Code

print("Following is the Depth-First Search")
dfs(visited, H, 0)            
#here bf search 
visited = [] # List to keep track of visited nodes.
queue = []     #Initialize a queue
def bfs(visited, graph, start, end):
  parent = {start: None}
  visited.append(start)
  queue.append(start)

  while queue:
    s = queue.pop(0) 

    for neighbour in graph[s]:
      if neighbour not in visited:
        parent[neighbour] = s
        if neighbour == end:
          path = []
          path.append(end)
          while parent[neighbour] != None:
            path.append(parent[neighbour])
            neighbour = parent[neighbour]
          return path[::-1]
        visited.append(neighbour)
        queue.append(neighbour)
        
# Driver Code
start = np.random.randint(0, 99)
end = np.random.randint(0, 99)

print("start node is {}, end node is {}".format(start, end))

print("shortest path is ", bfs(visited, H, start, end))

#plot it
color_map = []
for node in H:
    if node in [7, 51, 85, 78, 76, 50]:
        color_map.append('red')
    else: 
        color_map.append('grey') 

nx.draw(H, node_color=color_map, with_labels=True)
