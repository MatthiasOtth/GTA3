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
    def __init__(self, depth, seed=None):
        self.depth = depth
        self.seed = seed
        self.rng = np.random.RandomState(seed)


    def _num_tree_nodes(self):
        return 2 ** (self.depth + 1) - 1


    def _create_base_tree(self):
        num_nodes = self._num_tree_nodes()
        nodes = [0] * (num_nodes - 1) # tree nodes with type 0

        edges = list()
        for i in range(1, num_nodes):
            p = (i-1) >> 1
            edges.append((i,p))

        return num_nodes, nodes, edges


    def _add_neighbors(self, num_nodes, nodes, edges, root_neighbors, leaf_neighbors):
        for i in range(root_neighbors):
            edges.append((i+num_nodes, 0))

        num_nodes += root_neighbors
        leaf_base_idx = (2 ** self.depth) - 1
        for i in range(2 ** self.depth):
            for _ in range(leaf_neighbors[i]):
                edges.append((num_nodes, i+leaf_base_idx))
                num_nodes += 1
        
        nodes += [1] * (num_nodes - len(nodes))

        return num_nodes, nodes, edges


    def generate_data(self, train_size=0.8, valid_size=0.5, max_leaf_perm=1000, max_perm_examples=1000, max_examples=32000):
        num_leaf_nodes = 2 ** self.depth

        num_perm_examples = min(num_leaf_nodes, max_perm_examples)
        num_leaf_perm = min(max_leaf_perm, math.factorial(num_leaf_nodes), max_examples // num_perm_examples)
        permutations = [self.rng.permutation(range(num_leaf_nodes)) for _ in range(num_leaf_perm)]

        # generate graphs
        data_list = list()
        print(f"Generating {num_leaf_perm * num_perm_examples} graphs...")
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
                
                label = np.where(leaf_neighbors == root_neighbors)[0][0] + num_leaf_nodes - 1

                data_list.append((g, label))
        
        # split graphs into train, valid and test data
        train_data, test_data = train_test_split(data_list, train_size=train_size, shuffle=True, stratify=[l for _, l in data_list], random_state=self.seed)
        valid_data, test_data = train_test_split(test_data, train_size=valid_size, shuffle=False)

        # return (train_data, valid_data, test_data, num_types, num_tree_nodes)
        return train_data, valid_data, test_data, 2, self._num_tree_nodes()