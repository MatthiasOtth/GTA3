"""
File taken from 'On the Bottleneck of Graph Neural Networks and its Practical Implications' paper
by Uri Alon and Eran Yahav.
Link: https://github.com/tech-srl/bottleneck/blob/main/tasks/tree_dataset.py (19.12.2023)

Slightly adapted the output format to our applications.

"""

import torch
import dgl
import math
import numpy as np
from sklearn.model_selection import train_test_split


class TreeDataset():
    def __init__(self, depth, seed=None, directed=True):
        self.depth = depth
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.directed = directed


    def _num_tree_nodes(self):
        return 2 ** (self.depth + 1) - 1
    

    def _num_leaf_nodes(self):
        return 2 ** self.depth


    def _create_base_tree(self):
        num_nodes = self._num_tree_nodes()
        num_leaf_nodes = self._num_leaf_nodes()
        nodes = [[0,0]] * (num_nodes - num_leaf_nodes) + [[i,0] for i in range(2,num_leaf_nodes+2)]
        # nodes = [[0,0]]
        # for i in range(1, self.depth):
        #     nodes += [[i,0] for _ in range(2 ** i)]
        # nodes += [[i,0] for i in range(self.depth, num_leaf_nodes+self.depth)]

        edges = list()
        for i in range(1, num_nodes):
            p = (i-1) >> 1
            edges.append((i,p))
            if not self.directed: edges.append((p, i))

        return num_nodes, nodes, edges


    def _add_neighbors(self, num_nodes, nodes, edges, root_neighbors, leaf_neighbors):
        nodes[0][1] = root_neighbors

        leaf_base_idx = (2 ** self.depth) - 1
        for i in range(2 ** self.depth):
            edges.append((num_nodes, i+leaf_base_idx))
            if not self.directed:
                edges.append((i+leaf_base_idx, num_nodes))
            num_nodes += 1
        nodes += [[1,i] for i in leaf_neighbors]
        # neighbor_type = nodes[-1][0] + 1
        # nodes += [[neighbor_type,i] for i in leaf_neighbors]

        return num_nodes, nodes, edges


    def generate_data(self, train_size=0.8, valid_size=0.5, max_leaf_perm=1000, max_perm_examples=1000, max_examples=64000):
        num_leaf_nodes = 2 ** self.depth

        num_perm_examples = min(num_leaf_nodes, max_perm_examples)
        num_leaf_perm = min(max_leaf_perm, math.factorial(num_leaf_nodes), max_examples // num_perm_examples)
        permutations = [self.rng.permutation(range(num_leaf_nodes)) for _ in range(num_leaf_perm)]

        # generate graphs
        data_list = list()
        print(f"Generating {num_leaf_perm * num_perm_examples} graphs..." + ""*15)
        max_n = 0
        for perm in permutations:
            for i in self.rng.permutation(range(num_leaf_nodes))[:num_perm_examples]:

                leaf_neighbors = perm
                root_neighbors = i

                n, N, E = self._create_base_tree()
                n, N, E = self._add_neighbors(n, N, E, root_neighbors, leaf_neighbors)

                E = torch.tensor(E, dtype=torch.int).transpose(0,1)
                N = torch.tensor(N, dtype=torch.int)

                g = dgl.graph((E[0], E[1]), num_nodes=n)
                g.ndata['feat'] = N
                
                label = np.where(leaf_neighbors == root_neighbors)[0][0]

                # # DEBUG -> TODO: remove later
                # import networkx as nx
                # import matplotlib.pyplot as plt
                # g = nx.DiGraph()
                # feat = dict()
                # for i, n in enumerate(N):
                #     g.add_node(i)
                #     # feat[i] = int(n)
                #     feat[i] = f"type: {int(n[0])}, key: {int(n[1])}"
                # for i in range(E.size(1)):
                #     g.add_edge(int(E[0][i]), int(E[1][i]))
                # print(N)
                # print(label)
                # nx.draw_planar(g, arrows=True, labels=feat)
                # plt.show()
                # exit()

                data_list.append((g, label))
                max_n = max(max_n, n)
        
        # split graphs into train, valid and test data
        train_data, test_data = train_test_split(data_list, train_size=train_size, shuffle=True, stratify=[l for _, l in data_list], random_state=self.seed)
        valid_data, test_data = train_test_split(test_data, train_size=valid_size, shuffle=False)

        # return (train_data, valid_data, test_data, num_types, num_leaf_nodes)
        return train_data, valid_data, test_data, self._num_leaf_nodes()+2, self._num_leaf_nodes()
        # return train_data, valid_data, test_data, self._num_leaf_nodes()+self.depth+1, self._num_leaf_nodes()