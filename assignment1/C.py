import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph
import random


# Create G1 dataset
def create_dataset1():
    X = pd.read_table('Data_Highschool.txt')
    dataset1 = np.array(X)
    return dataset1


# Create G2 dataset
def create_dataset2():
    X = pd.read_table('Data_Highschool.txt')
    dataset2 = np.array(X)
    dataset2_timelist = dataset2[:, 2].tolist()
    random.shuffle(dataset2_timelist)
    dataset2[:, 2] = np.array(dataset2_timelist)
    dataset2 = dataset2[np.lexsort(dataset2.T)]
    return dataset2


# Create G3 dataset
def create_dataset3():
    X = pd.read_table('Data_Highschool.txt')
    dataset = np.array(X)
    dataset3 = np.insert(dataset[:, 0:2], 2, values=np.array([0 for x in range(1, len(dataset) + 1)]), axis=1)
    dataset3_timelist = dataset[:, 2].tolist()
    for i in dataset3_timelist:
        n = random.randint(0, len(dataset3) - 1)
        if dataset3[n, 2] == 0:
            dataset3[n, 2] = i
    drow = np.argwhere(dataset3 == 0)[:, 0]
    dataset3 = np.delete(dataset3, drow, 0)
    dataset3 = dataset3[np.lexsort(dataset3.T)]
    return dataset3


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

# Create dataset
dataset1 = create_dataset1()
dataset2 = create_dataset2()
dataset3 = create_dataset3()

dataset = dataset3

# create node list, time list and node, link dictionary each time step
node_list = sorted(np.unique(dataset[:, 0].tolist() + dataset[:, 1].tolist()))
time_list = sorted(np.unique(dataset1[:, 2]).tolist())
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
print(seedall_dic)


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
plt.title('G3')
plt.show()


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
print(sorted(R, key=lambda x: x[1]))


