import torch
from torch.utils.data import Dataset
import os
from dgl import shortest_dist


def transform_to_graph_list(dataset):
    g_list = list()
    for g in dataset:
        g_list.append(g)
    return g_list


class GTA3BaseDataset(Dataset):

    def __init__(self, name, mode, phi_func, force_reload=False, compute_class_weights=False):
        self.num_classes = None
        self.compute_class_weights = compute_class_weights
        path = './.dgl'
        self.data_path = None

        # determine necessary precomputation steps
        self.use_shortest_dist = False
        self.use_adj_matrix = False
        if phi_func == 'test':
            self.use_adj_matrix = True
        elif phi_func == 'inverse_hops':
            self.use_shortest_dist = True
        else:
            raise ValueError(f"Invalid value for phi_func: '{phi_func}'!")

        # define caching paths
        if self.use_adj_matrix or self.use_shortest_dist or compute_class_weights:
            path = os.path.join(path, name, mode)
            if self.use_adj_matrix: path += '_adj'
            if self.use_shortest_dist: path += '_sd'
            data_path = path + '_data.bin'

            if self.compute_class_weights: path += '_cw'
            info_path = path + '_info.pkl'

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
        num_graphs = len(self.graphs)

        # adjacency matrix and shortest path matrix
        if self.use_adj_matrix:
            print(f"Precomputing adjacency matrix (0/{num_graphs})...", end="\r")
            for idx, g in enumerate(self.graphs):
                # TODO: this is horrible I know but it works for now...
                adj_mat = torch.zeros((g.num_nodes(), g.num_nodes()))
                u, v = g.edges()
                for i in range(len(u)):
                    adj_mat[u[i]][v[i]] = 1
                #adj_mat = F.softmax(adj_mat, dim=1) # TODO: tryout
                if self.use_adj_matrix:
                    g.ndata['adj_mat'] = adj_mat

                if idx % 50 == 0:
                    print(f"Precomputing adjacency matrix ({idx}/{num_graphs})...", end="\r")
            print(f"Precomputing adjacency matrix ({num_graphs}/{num_graphs})...", end="\r")

        # create shortest distant matrix
        if self.use_shortest_dist:
            print(f"Precomputing shortest distance matrix (0/{num_graphs})...", end="\r")
            for idx, g in enumerate(self.graphs):
                sd_mat = shortest_dist(g, return_paths=False)
                for i in range(g.num_nodes()):
                    sd_mat[i][i] = 1 # set distance to self to 1 (equal to one message passing hop)
                g.ndata['short_dist_mat'] = sd_mat

                if idx % 50 == 0:
                    print(f"Precomputing shortest distance matrix ({idx}/{num_graphs})...", end="\r")
            print(f"Precomputing shortest distance matrix ({num_graphs}/{num_graphs})...", end="\r")

                # short_dist = nx.shortest_path_length(nx.from_numpy_array(adj_mat.numpy(),create_using=nx.DiGraph))
                # short_dist_mat = torch.zeros_like(adj_mat)
                # for i, j_d in short_dist:
                #     for j, d in j_d.items():
                #         short_dist_mat[i, j] = d
                #     # Dist i,i is 0, but we want it to be 1, since it takes 1 MessagePassing hop
                #     short_dist_mat[i, i] = 1
                # g.ndata['short_dist_mat'] = short_dist_mat

        # compute class weights
        # > requires self.num_classes to not be None
        if self.compute_class_weights:
            assert self.num_classes is not None, "GTA3BaseDataset: Define self.num_classes to compute the class weights!"
            print(f"Precomputing class weights (0/{num_graphs})...           ", end="\r")

            self.class_weights = list()
            for idx, g in enumerate(self.graphs):
                num_nodes = g.num_nodes()
                labels_counted = torch.bincount(g.ndata['label'])
                labels_counted_nz = labels_counted.nonzero().squeeze()
                class_sizes = torch.zeros(self.num_classes, dtype=torch.long)
                class_sizes[labels_counted_nz] = labels_counted[labels_counted_nz]
                weights = (num_nodes - class_sizes).to(dtype=torch.float) / num_nodes
                weights = torch.where(class_sizes > 0, weights, torch.zeros_like(weights))
                self.class_weights.append(weights)

                if idx % 50 == 0:
                    print(f"Precomputing class weights ({idx}/{num_graphs})...", end="\r")
            print(f"Precomputing class weights ({num_graphs}/{num_graphs})...", end="\r")