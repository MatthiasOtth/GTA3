"""
File taken from 'On the Bottleneck of Graph Neural Networks and its Practical Implications' paper
by Uri Alon and Eran Yahav.
Link: https://github.com/tech-srl/bottleneck/blob/main/tasks/tree_dataset.py (19.12.2023)

Slightly adapted the output format to our applications.

"""

import torch
import dgl

from torch.nn import functional as F
from sklearn.model_selection import train_test_split


class TreeDataset(object):
    def __init__(self, depth, seed=None):
        super(TreeDataset, self).__init__()
        self.depth = depth
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.seed = seed

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self):
        edge_index = torch.tensor(self.edges).t()
        return edge_index

    def generate_data(self, train_size, test_size):
        # NOTE: first test size is taken from the data, then the rest is split according to the train_size
        data_list = []

        for comb in self.get_combinations():
            edge_index = self.create_blank_tree()
            nodes = torch.tensor(self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1), dtype=torch.bool)
            label = self.label(comb)
            
            g = dgl.graph((edge_index[0], edge_index[1]))
            g.ndata['feat'] = nodes
            g.ndata['root_mask'] = root_mask
            data_list.append((g, label))

        dim0, out_dim = self.get_dims()
        
        if self.seed is not None:
            test_data, train_data = train_test_split(data_list, train_size=test_size, shuffle=True, stratify=[l for _, l in data_list], random_state=self.seed)
            train_data, valid_data = train_test_split(data_list, train_size=train_size, shuffle=False, random_state=self.seed)
        else:
            test_data, train_data = train_test_split(data_list, train_size=test_size, shuffle=True, stratify=[l for _, l in data_list])
            train_data, valid_data = train_test_split(data_list, train_size=train_size, shuffle=False)

        return train_data, valid_data, test_data, dim0, out_dim

    # Every sub-class should implement the following methods:
    def get_combinations(self):
        raise NotImplementedError

    def get_nodes_features(self, combination):
        raise NotImplementedError

    def label(self, combination):
        raise NotImplementedError

    def get_dims(self):
        raise NotImplementedError