import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import igraph

# read file
dataset = pd.read_table('Data_Highschool.txt')

# create node list and link list
node_list = sorted(np.unique(np.array(dataset.loc[:, ['i']]).tolist() + np.array(dataset.loc[:, ['j']]).tolist()))
link_list = []
for i in np.array(dataset.loc[:, ['i', 'j']]).tolist():
    if i not in link_list:
        link_list.append(i)
link_list = sorted(link_list, key=lambda x: x[0])
#print(node_list)
#print(link_list)

# create graph
G = nx.Graph(link_list)

'''
# Question 2
degree = nx.degree_histogram(G)
x = range(len(degree))
y = [n / float(sum(degree)) for n in degree]

# log y = a * x + b
plt.figure()
plt.semilogy(x, y)
plt.title('log y = a * x + b')
plt.xlabel('k')
plt.ylabel('log P(k)')


# log y = a * log x + b
plt.figure()
plt.loglog(x, y)
plt.title('log y = a * log x + b')
plt.xlabel('log k')
plt.ylabel('log P(k)')

# y = a * x + b
plt.figure()
plt.plot(x, y)
plt.show()
plt.title('y = a * x + b')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.show()


# Question 3
deg_correlation = nx.degree_assortativity_coefficient(G)
print(deg_correlation)


# Question 4
cluster_coefficient = nx.average_clustering(G)
print(cluster_coefficient)


# Question 5
avg_hopcount = nx.average_shortest_path_length(G)
print(avg_hopcount)
diameter = nx.diameter(G)
print(diameter)


# Question 7
adj_matrix = nx.to_numpy_matrix(G)
eig_values = np.max(np.linalg.eigvals(adj_matrix))
print(adj_matrix.shape)
'''

# Question 8
alg_connectivity = nx.algebraic_connectivity(G)
print(alg_connectivity)
