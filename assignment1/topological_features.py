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
G = igraph.Graph(link_list)
'''
visual_style = {}
visual_style["vertex_label_size"]=1
visual_style["vertex_label"] = node_list
visual_style["edge_label"] = link_list
visual_style["layout"] = G.layout("kk")
igraph.plot(G, **visual_style)


# Question 1
node_num = G.vcount()
link_num = G.ecount()
density = igraph.Graph.density(G)
avg_degree = 2 * link_num / node_num
var_degree = np.var(np.asarray(G.degree()))
print('Number of nodes:', node_num)
print('Number of link:', link_num)
print('Link density:', density)
print('Average degree:', avg_degree)
print('Degree variance:', var_degree)


# Question 2
degree = igraph.Graph.degree(G)
degree_dic = {}
for i in degree:
    if i not in degree_dic:
        degree_dic[i] = 1
    else:
        degree_dic[i] += 1
for n in range(max(degree_dic.keys())):
    if n not in degree_dic.keys():
        degree_dic[n] = 0
x = range(len(degree_dic.values()))
y = [degree_dic[n] / float(sum(degree_dic.values())) for n in x]

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
plt.title('y = a * x + b')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.show()


# Question 3
degree_correl = G.assortativity_degree()
print(degree_correl)


# Question 4
cluster_coefficient = G.transitivity_undirected(mode="nan")
print(cluster_coefficient)


# Question 5
avg_hopcount = G.average_path_length()
diameter = G.diameter()
print(avg_hopcount)
print(diameter)

'''
# Question 7
adj_matrix = np.array(G.get_adjacency().data)
eig_values = np.max(np.linalg.eigvals(adj_matrix))
print(eig_values)



# Question 8
lap_matrix = G.laplacian()
eig_vector = np.sort(np.linalg.eigvals(lap_matrix))
alg_connectivity = eig_vector[1]
print(eig_vector)
