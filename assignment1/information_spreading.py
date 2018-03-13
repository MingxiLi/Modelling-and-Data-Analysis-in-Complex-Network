import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph


# Create node and link dictionary each time step
def create_nl(d):
    node = {}
    net = {}

    for data in d:
        time = data[2]
        node[time] = []

    for data in d:
        node1 = data[0]
        node2 = data[1]
        time = data[2]

        if node1 not in node[time]:
            node[time].append(node1)
        if node2 not in node[time]:
            node[time].append(node2)

        if time in net:
            net[time].append([node1, node2])
        else:
            net[time] = []
            net[time].append([node1, node2])
    return node, net


def simulate(nodes, links, node_num, endtime, seed):

    # total infected nodes
    node_infected = [seed]
    # total infected nodes in each time step
    time_infnode = {}

    for t in range(1, endtime + 1):
        if t not in links:
            time_infnode[t] = len(node_infected)
        elif len(node_infected) == node_num:
            time_infnode[t] = len(node_infected)
        else:
            # Create graph at time t
            Gt = igraph.Graph(links[t])
            # all nodes connected at time t
            for node in nodes[t]:
                # if a node is infected, its neighbours will be infected
                if node in node_infected:
                    for n in Gt.neighbors(node):
                        if n not in node_infected:
                            node_infected.append(n)
            time_infnode[t] = len(node_infected)
    #print(time_infnode)
    return time_infnode


####################################################

# read file
X = pd.read_table('Data_Highschool.txt')
dataset = np.array(X)

# create node list, time list and node, link dictionary each time step
node_list = sorted(np.unique(dataset[:, 0].tolist() + dataset[:, 1].tolist()))
time_list = sorted(np.unique(dataset[:, 2]).tolist())
T = max(time_list)
node_dic_time, link_dic_time = create_nl(dataset)
link_list = dataset[:, 0:2].tolist()
#print(sorted(node_dic_time[1]))
#print(sorted(link_dic_time[1]))


# Dictionary with how infected nodes change with different seeds
seedall_dic = {}

# total number of nodes N
N = len(node_list)

# seed of the information
for seed in range(1, N + 1):
    inf = simulate(node_dic_time, link_dic_time, max(node_list), T, seed)
    #print(inf)
    if seed in seedall_dic.keys():
        seedall_dic[seed].append(inf)
    else:
        seedall_dic[seed] = inf
#print(seedall_dic)

'''
###### Question 9 ######
avg = []
var = []
# All infected nodes each time step for N iterations
node_all = {}
for seed in seedall_dic.keys():
    for t in seedall_dic[seed].keys():
        if t in node_all.keys():
            node_all[t].append(seedall_dic[seed][t])
        else:
            node_all[t] = []
            node_all[t].append(seedall_dic[seed][t])
#print(node_all)
for t in node_all.keys():
    avg.append(np.mean(node_all[t]))
    var.append(np.std(node_all[t]))
#print(avg)
#print(var)
plt.plot(time_list, avg)
plt.plot(time_list, var)
plt.annotate('average value', xy=(3000, 320), xytext=(4000, 250), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
plt.annotate('standard deviation', xy=(6000, 10), xytext=(4000, 100), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
plt.xlabel('Time step t')
plt.show()
'''

###### Question 10 ######
threshold = 0.8 * N
R = []  #[(seed, time)]
for n in seedall_dic.keys():
    for t in seedall_dic[n].keys():
        if seedall_dic[n][t] >= threshold:
            R.append((n, t))
            break
R_vector = [tup[0] for tup in sorted(R, key=lambda x: x[1])]
print(R_vector)


'''
###### Question 11 ######
degree_list = []  #(node, degree)
cc_list = []  #(node, clustering coefficient)
G = igraph.Graph(link_list)
for n in node_list:
    degree_list.append((n, G.degree(n)))
    cc_list.append((n, G.transitivity_local_undirected(n)))
# Ordered degree vector
D_vector = [tup[0] for tup in sorted(degree_list, key=lambda x: x[1])]
#print(D_vector)
# Ordered clustering coefficient vector
CC_vector = [tup[0] for tup in sorted(cc_list, key=lambda x: x[1])]
#print(CC_vector)
F = []
for f in np.arange(0.05, 0.55, 0.05):
    F.append(f)
# rRDf
rRDf = []
for f in F:
    length = int(f * len(D_vector))
    #print(f)
    rRDf.append(float(len(list(set(R_vector[: length]).intersection(set(D_vector[: length])))) / float(len(R_vector[: length]))))
    #print(rRDf[: length])
# rRCf
rRCf = []
for f in F:
    length = int(f * len(CC_vector))
    #print(f)
    rRCf.append(float(len(list(set(R_vector[: length]).intersection(set(CC_vector[: length])))) / float(len(R_vector[: length]))))
    #print(rRCf[: length])
plt.plot(F, rRDf, 'b')
plt.plot(F, rRCf, 'r')
plt.xlabel('f')
plt.show()
'''
between_list = []  #(node, betweeness)
clc_list = []  #(node, closeness)
G = igraph.Graph(link_list)
for n in node_list:
    between_list.append((n, G.betweenness(n)))
    clc_list.append((n, G.closeness(n)))
# Ordered betweenness vector
B_vector = [tup[0] for tup in sorted(between_list, key=lambda x: x[1])]
# Ordered closeness vector
C_vector = [tup[0] for tup in sorted(clc_list, key=lambda x: x[1])]

F = []
for f in np.arange(0.05, 0.55, 0.05):
    F.append(f)
# rRDf
rRBf = []
for f in F:
    length = int(f * len(B_vector))
    #print(f)
    rRBf.append(float(len(list(set(R_vector[: length]).intersection(set(B_vector[: length])))) / float(len(R_vector[: length]))))
    #print(rRDf[: length])
# rRCf
rRCf = []
for f in F:
    length = int(f * len(C_vector))
    #print(f)
    rRCf.append(float(len(list(set(R_vector[: length]).intersection(set(C_vector[: length])))) / float(len(R_vector[: length]))))
    #print(rRCf[: length])
plt.plot(F, rRBf, 'b')
plt.plot(F, rRCf, 'r')
plt.xlabel('f')
plt.show()
