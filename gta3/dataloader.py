import torch
from torch.utils.data import Dataset
import os
from dgl import shortest_dist

from gta3.pos_encoder import laplacian_positional_encoding, wl_positional_encoding


def transform_to_graph_list(dataset):
    g_list = list()
    for g in dataset:
        g_list.append(g)
    return g_list


class GTA3BaseDataset(Dataset):

    def __init__(self, name, mode, phi_func, pos_enc, batch_size=10, force_reload=False, pos_enc_dim=8, compute_class_weights=False, path_suffix=''):
        self.pos_enc_dim = pos_enc_dim
        self.compute_class_weights = compute_class_weights
        self.batch_size = batch_size
        self.num_classes = None
        
        path = './.dgl'
        self.data_path = None

        # determine necessary precomputation steps
        # > phi function
        self.use_shortest_dist = False
        self.use_adj_matrix = False
        if phi_func in ['test', 'local']:
            self.use_adj_matrix = True
        elif (
            phi_func in 
                [
                    'inverse_hops',
                    'inverse_hops_exp',
                    'alpha_pow_dist',
                    'alpha_pow_dist_exp',
                    'alpha_pow_dist_sigmoid',
                    'alpha_pow_dist_sigmoid_softmax',
                    'phi_poisson_exp',
                    'gaussian_std1',
                    'sigmoid_shift'
                ]
        ):
            self.use_shortest_dist = True
        elif phi_func != 'none':
            raise ValueError(f"GTA3BaseDataset Error: Invalid value for phi_func: '{phi_func}'!")
        
        # > positional encoding
        self.use_lap_pos_enc = False
        self.use_wl_pos_enc = False
        if pos_enc == 'laplacian':
            self.use_lap_pos_enc = True
        elif pos_enc == 'wl':
            self.use_wl_pos_enc = True
        elif pos_enc != 'none':
            raise ValueError(f"GTA3BaseDataset Error: Invalid value for pos_enc: '{pos_enc}'!")

        # define caching paths
        if self.use_adj_matrix or self.use_shortest_dist or compute_class_weights:
            path = os.path.join(path, name, mode)
            if self.use_adj_matrix: path += '_adj'
            if self.use_shortest_dist: path += '_sd'
            if self.use_lap_pos_enc: path += '_lap'
            if self.use_wl_pos_enc: path += '_wl'
            data_path = path + f'_data{path_suffix}.bin'

            if self.compute_class_weights: path += '_cw'
            info_path = path + f'_info{path_suffix}.pkl'

        # load data
        # > load preprocessed data if it exists
        if not force_reload and data_path is not None and os.path.exists(data_path) and os.path.exists(info_path):
            self._load_cached_data(data_path, info_path)
            self.num_graphs = len(self.graphs)

        # > load raw data and preprocess it
        else:
            self._load_raw_data(data_path, info_path)

        # batch data
        if batch_size > 1:
            self._create_batches()

        # define data length
        self.data_len = len(self.graphs)
        if batch_size > 1:
            self.data_len = self.num_batches


    def __len__(self):
        return self.data_len


    def __getitem__(self, idx):

        # batched output
        if self.batch_size > 1:

            # > including class weights
            if self.compute_class_weights:
                return (
                    self.batches[idx][0], 
                    self.batches[idx][1], 
                    self.batches[idx][2], 
                    self.batches[idx][3], 
                    self.batches[idx][4], 
                    self.batches[idx][5])
            
            # > without class weights
            return (
                self.batches[idx][0], 
                self.batches[idx][1], 
                self.batches[idx][2], 
                self.batches[idx][3], 
                self.batches[idx][4])
        
        # single graph output
        # > pick phi matrix (adjacency matrix, shortest path or none)
        if self.use_adj_matrix:
            phi_mat = self.graphs[idx].ndata['adj_mat']
        elif self.use_shortest_dist:
            phi_mat = self.graphs[idx].ndata['short_dist_mat']
        else:
            phi_mat = None

        # > pick pos enc (laplacian, wl-based or none)
        if self.use_lap_pos_enc:
            pos_enc = self.graphs[idx].ndata['lap_pos_enc']
        elif self.use_wl_pos_enc:
            pos_enc = self.graphs[idx].ndata['wl_pos_enc']
        else:
            pos_enc = None

        # > including class weights
        if self.compute_class_weights:
            return (
                self.graphs[idx].num_nodes(),
                self.graphs[idx].ndata['feat'], 
                phi_mat, 
                pos_enc, 
                self._get_label(idx), 
                self.class_weights[idx])
        
        # > without class weights
        return (
            self.graphs[idx].num_nodes(),
            self.graphs[idx].ndata['feat'], 
            phi_mat, 
            pos_enc, 
            self._get_label(idx))


    def _load_raw_data(self, data_path, info_path):
        raise NotImplementedError("GTA3BaseDataset: Implement the _load_raw_data function!")
    

    def _load_cached_data(self, data_path, info_path):
        raise NotImplementedError("GTA3BaseDataset: Implement the _load_cached_data function!")


    def _get_label(self, idx):
        raise NotImplementedError("GTA3BaseDataset: Implement the _get_label function!")


    def _preprocess_data(self):
        self.num_graphs = len(self.graphs)

        # add adjacency matrix
        if self.use_adj_matrix:
            print(f"Precomputing adjacency matrix (0/{self.num_graphs})...", end="\r")
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
                    print(f"Precomputing adjacency matrix ({idx}/{self.num_graphs})...", end="\r")
            print(f"Precomputing adjacency matrix ({self.num_graphs}/{self.num_graphs})...", end="\r")

        # add shortest distant matrix
        if self.use_shortest_dist:
            print(f"Precomputing shortest distance matrix (0/{self.num_graphs})...", end="\r")
            for idx, g in enumerate(self.graphs):
                sd_mat = shortest_dist(g, return_paths=False)
                for i in range(g.num_nodes()):
                    sd_mat[i][i] = 1 # set distance to self to 1 (equal to one message passing hop)
                sd_mat = torch.where(sd_mat==-1, torch.zeros_like(sd_mat), sd_mat)
                g.ndata['short_dist_mat'] = sd_mat

                if idx % 50 == 0:
                    print(f"Precomputing shortest distance matrix ({idx}/{self.num_graphs})...", end="\r")
            print(f"Precomputing shortest distance matrix ({self.num_graphs}/{self.num_graphs})...", end="\r")

        # add laplacian positional encodings
        if self.use_lap_pos_enc:
            print(f"Precomputing laplacian positional encodings (0/{self.num_graphs})...", end="\r")
            assert self.pos_enc_dim is not None, "GTA3BaseDataset: pos_enc_dim is None!"
            for idx, g in enumerate(self.graphs):
                laplacian_positional_encoding(g, self.pos_enc_dim)

                if idx % 50 == 0:
                    print(f"Precomputing laplacian positional encodings ({idx}/{self.num_graphs})...", end="\r")
            print(f"Precomputing laplacian positional encodings ({self.num_graphs}/{self.num_graphs})...", end="\r")

        # add wl-based positional embeddings
        if self.use_wl_pos_enc:
            print(f"Precomputing wl-based positional encodings (0/{self.num_graphs})...", end="\r")
            for idx, g in enumerate(self.graphs):
                wl_positional_encoding(g)

                if idx % 50 == 0:
                    print(f"Precomputing wl-based positional encodings ({idx}/{self.num_graphs})...", end="\r")
            print(f"Precomputing wl-based positional encodings ({self.num_graphs}/{self.num_graphs})...", end="\r")

        # compute class weights
        # > requires self.num_classes to not be None
        if self.compute_class_weights:
            assert self.num_classes is not None, "GTA3BaseDataset: Define self.num_classes to compute the class weights!"
            print(f"Precomputing class weights (0/{self.num_graphs})...           ", end="\r")

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
                    print(f"Precomputing class weights ({idx}/{self.num_graphs})...", end="\r")
            print(f"Precomputing class weights ({self.num_graphs}/{self.num_graphs})...", end="\r")


    def _create_batches(self):

        # get the label size and additional feature dimension
        label_size = self._get_label(0).size(0)
        if len(self.graphs[0].ndata['feat'].shape) > 1:
            feat_dim = self.graphs[0].ndata['feat'].size(1)
        else:
            feat_dim = 0

        # pad graphs
        print(f"Creating batches (0/{self.num_graphs})..." + ' '*15, end="\r")
        self.num_batches = (self.num_graphs + 1) // self.batch_size
        self.batches = list()

        for batch_idx in range(self.num_batches):
            graph_base_idx = batch_idx * self.batch_size
            graph_top_idx = graph_base_idx + self.batch_size
            graph_top_idx = graph_top_idx if graph_top_idx < self.num_graphs else self.num_graphs
            curr_batch_size = graph_top_idx - graph_base_idx

            # get maximum number of nodes in this batch
            max_num_nodes = 0
            for g in self.graphs[graph_base_idx:graph_top_idx]:
                n = g.num_nodes()
                if n > max_num_nodes:
                    max_num_nodes = n

            # init new batch tensors
            batch_num_nodes = torch.zeros((curr_batch_size), dtype=torch.int) # TODO: dtype
            batch_feat_shape = (curr_batch_size, max_num_nodes) if feat_dim == 0 else (curr_batch_size, max_num_nodes, feat_dim)
            batch_feat = torch.zeros(batch_feat_shape, dtype=torch.int) # TODO: dtype
            if self.use_adj_matrix or self.use_shortest_dist:
                batch_phi_mat = torch.zeros((curr_batch_size, max_num_nodes, max_num_nodes), dtype=torch.int) # TODO: dtype
            else:
                batch_phi_mat = None
            if self.use_lap_pos_enc or self.use_wl_pos_enc:
                if self.use_lap_pos_enc:
                    batch_pos_enc = torch.zeros((curr_batch_size, max_num_nodes, self.pos_enc_dim))
                else:
                    batch_pos_enc = torch.zeros((curr_batch_size, max_num_nodes), dtype=torch.long)
            else:
                batch_pos_enc = None
            
            # process graphs
            for idx, g in enumerate(self.graphs[graph_base_idx:graph_top_idx]):
                
                # > save number of nodes
                batch_num_nodes[idx] = g.num_nodes()

                # > pad node features
                batch_feat[idx, :g.num_nodes()] = g.ndata['feat']
                batch_feat[idx, g.num_nodes():] = self.get_num_in_types() - 1 # Take last type, should be unused
                
                # > pad adjacency matrix
                if self.use_adj_matrix:
                    batch_phi_mat[idx, :g.num_nodes(), :g.num_nodes()] = g.ndata['adj_mat']

                # > pad shortest distance matrix
                elif self.use_shortest_dist:
                    batch_phi_mat[idx, :g.num_nodes(), :g.num_nodes()] = g.ndata['short_dist_mat']

                # > laplacian positional encodings
                if self.use_lap_pos_enc:
                    batch_pos_enc[idx, :g.num_nodes()] = g.ndata['lap_pos_enc']
                
                # > wl-based positional encodings
                elif self.use_wl_pos_enc:
                    batch_pos_enc[idx, :g.num_nodes()] = g.ndata['wl_pos_enc']

            # process labels
            if label_size > 1:
                label_size = max_num_nodes
            batch_label = torch.zeros((curr_batch_size, label_size)) # TODO: dtype
            for idx in range(graph_top_idx - graph_base_idx):
                label = self._get_label(idx + graph_base_idx)
                if label.size(0) == 1:
                    batch_label[idx] = label
                else:
                    batch_label[idx, :label.size(0)] = label

            # process class weights and add batch to batches list
            if self.compute_class_weights:
                batch_class_weights = torch.concat([x.unsqueeze(0) for x in self.class_weights[graph_base_idx:graph_top_idx]], dim=0)
                batch_class_weights = torch.mean(batch_class_weights, dim=0)
                self.batches.append((batch_num_nodes, batch_feat, batch_phi_mat, batch_pos_enc, batch_label, batch_class_weights))
            else:
                self.batches.append((batch_num_nodes, batch_feat, batch_phi_mat, batch_pos_enc, batch_label))

            print(f"Creating batches ({graph_top_idx}/{self.num_graphs})...", end="\r")
        print(f"Creating batches ({self.num_graphs}/{self.num_graphs})...Done")
    
            
    def get_num_types(self):
        raise NotImplementedError("GTA3BaseDataset: Implement the get_num_types function!")
    

    def get_num_in_types(self):
        if self.batch_size > 1:
            return self.get_num_types() + 1
        else:
            return self.get_num_types()
        

    def get_num_out_types(self):
        return self.get_num_types()