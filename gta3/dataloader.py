import torch
from torch.utils.data import Dataset
import networkx as nx
import os


class GTA3BaseDataset(Dataset):

    def __init__(self, name, mode, phi_func, force_reload=False, compute_class_weights=False):
        self.num_classes = None
        self.compute_class_weights = None
        path = './.dgl'

        # determine necessary precomputation steps
        self.use_shortest_path = False
        self.use_adj_matrix = False
        data_path = None
        if phi_func == 'test':
            self.use_adj_matrix = True
            data_path = os.path.join(path, name, f'{mode}_adj.bin')
            info_path = os.path.join(path, name, f'{mode}_adj_info.pkl')
        elif phi_func == 'inverse_hops':
            self.use_shortest_path = True
            data_path = os.path.join(path, name, f'{mode}_adj_sp.bin')
            info_path = os.path.join(path, name, f'{mode}_adj_sp_info.pkl')

        # load data
        # > load preprocessed data if it exists
        if not force_reload and data_path is not None and os.path.exists(data_path) and os.path.exists(info_path):
            self._load_cached_data(data_path, info_path)

        # > load raw data and preprocess it
        else:
            self._load_raw_data(data_path, info_path)


    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, idx):
        raise NotImplementedError("GTA3BaseDataset: Implement the __getitem__ function!")


    def _load_raw_data(self, data_path, info_path):
        raise NotImplementedError("GTA3BaseDataset: Implement the _load_raw_data function!")
    

    def _load_cached_data(self, data_path, info_path):
        raise NotImplementedError("GTA3BaseDataset: Implement the _load_cached_data function!")


    def _preprocess_data(self):

        # adjacency matrix and shortest path matrix
        if self.use_adj_matrix or self.use_shortest_path or self.compute_class_weights:
            for g in self.graphs:
                
                # create the adjacency matrix for each graph
                # TODO: this is horrable I know but it works for now...
                adj_mat = torch.zeros((g.num_nodes(), g.num_nodes()))
                u, v = g.edges()
                for i in range(len(u)):
                    adj_mat[u[i]][v[i]] = 1
                #adj_mat = F.softmax(adj_mat, dim=1) # TODO: tryout
                if self.use_adj_matrix:
                    g.ndata['adj_mat'] = adj_mat

                # create shortest path matrix
                if self.use_shortest_path:
                    short_dist = nx.shortest_path_length(nx.from_numpy_array(adj_mat.numpy(),create_using=nx.DiGraph))
                    short_dist_mat = torch.zeros_like(adj_mat)
                    for i, j_d in short_dist:
                        for j, d in j_d.items():
                            short_dist_mat[i, j] = d
                        # Dist i,i is 0, but we want it to be 1, since it takes 1 MessagePassing hop
                        short_dist_mat[i, i] = 1
                    g.ndata['short_dist_mat'] = short_dist_mat

        # compute class weights
        # > requires self.num_classes to not be None
        if self.compute_class_weights:
            assert self.num_classes is not None, "GTA3BaseDataset: Define self.num_classes to compute the class weights!"
            self.class_weights = list()

            for g in self.graphs:

                num_nodes = g.num_nodes()
                print(num_nodes)

                labels_counted = torch.bincount(g.ndata['label'])
                labels_counted_nz = labels_counted.nonzero().squeeze()
                print(labels_counted.nonzero().squeeze())

                class_sizes = torch.zeros(self.num_classes, dtype=torch.long)
                print(class_sizes)

                class_sizes[labels_counted_nz] = labels_counted[labels_counted_nz]
                print(class_sizes)

                weights = (num_nodes - class_sizes).to(dtype=torch.float) / num_nodes
                weights = torch.where(class_sizes > 0, weights, torch.zeros_like(weights))
                print(weights)

                g.ndata['class_weights'] = weights
                exit()