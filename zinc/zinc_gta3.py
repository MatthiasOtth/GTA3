import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset
import networkx as nx
from dgl import save_graphs, load_graphs
from dgl.data import ZINCDataset
from dgl.data.utils import save_info, load_info
import os

from gta3.model import GTA3BaseModel
from gta3.loss import L1Loss_L1Alpha, L1Loss_L2Alpha


class GTA3_ZINC_Dataset(Dataset):

    def __init__(self, mode, phi_func, force_reload=False):
        path = './.dgl/'

        # determine necessary precomputation steps
        self.use_shortest_path = False
        self.use_adj_matrix = False
        data_path = None
        if phi_func == 'test':
            self.use_adj_matrix = True
            data_path = path + f'{mode}_adj.bin'
            info_path = path + f'{mode}_adj_info.pkl'
        elif phi_func == 'inverse_hops':
            self.use_shortest_path = True
            data_path = path + f'{mode}_adj_sp.bin'
            info_path = path + f'{mode}_adj_sp_info.pkl'

        # load data
        # > load preprocessed data if it exists
        if not force_reload and data_path is not None and os.path.exists(data_path) and os.path.exists(info_path):
            print(f"Loading cached ZINC {mode} data...", end='\r')
            self.graphs, labels_dict = load_graphs(data_path)
            self.labels = labels_dict['labels']
            self.num_atom_types = load_info(info_path)['num_atom_types']
            print(f"Loading cached ZINC {mode} data...Done")

        # > load raw data and preprocess it
        else:

            # load raw data
            print(f"Loading the raw ZINC {mode} data...", end='\r')
            raw_data = ZINCDataset(mode=mode, raw_dir='./.dgl/')
            self.graphs = raw_data[:][0]
            self.labels = raw_data[:][1]
            self.num_atom_types = raw_data.num_atom_types
            print(f"Loading the raw ZINC {mode} data.......Done")

            # preprocess data
            print(f"Preprocessing the {mode} data...", end='\r')
            self._preprocess_data()
            print(f"Preprocessing the {mode} data..........Done")

            # store the preprocessed data
            print(f"Caching the preprocessed {mode} data...", end='\r')
            save_graphs(data_path, self.graphs, {'labels': self.labels})
            save_info(info_path, {'num_atom_types': self.num_atom_types})
            print(f"Caching the preprocessed {mode} data...Done")


    def __len__(self):
        return len(self.graphs)


    def __getitem__(self, idx):
        # return (node features, adjacency matrix, label) tuple
        if self.use_adj_matrix:
            return self.graphs[idx].ndata['feat'], self.graphs[idx].ndata['adj_mat'], self.labels[idx].unsqueeze(0)
        elif self.use_shortest_path:
            return self.graphs[idx].ndata['feat'], self.graphs[idx].ndata['short_dist_mat'], self.labels[idx].unsqueeze(0)
        else:
            return self.graphs[idx].ndata['feat'], None, self.labels[idx].unsqueeze(0)


    def _preprocess_data(self):

        if self.use_adj_matrix or self.use_shortest_path:
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

    
    def get_num_types(self):
        return self.num_atom_types

        


class GTA3_ZINC(GTA3BaseModel):
    
    def __init__(self, model_params, train_params):
        
        # initialize the GTA3 base model
        super().__init__(model_params, train_params)

        # final mlp to map the out dimension to a single value
        self.out_mlp = nn.Sequential(nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2), nn.ReLU(), nn.Dropout(), nn.Linear(model_params['out_dim'] * 2, 1))
        # self.out_mlp = nn.Sequential(
        #     nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2),
        #     nn.ReLU(), 
        #     # nn.Dropout(), 
        #     nn.Linear(model_params['out_dim'] * 2, model_params['out_dim'] // 2),
        #     nn.ReLU(), 
        #     # nn.Dropout(), 
        #     nn.Linear(model_params['out_dim'] // 2, 1),
        # )
        
        # loss functions
        if model_params['alpha'] == 'fixed':
            self.train_alpha = False
            self.train_loss_func = nn.L1Loss()
        else:
            self.train_alpha = True
            self.train_loss_func = L1Loss_L1Alpha
            self.alpha_weight = model_params['alpha_weight']
        self.valid_loss_func = nn.L1Loss()


    def forward_step(self, x, A):
        self.alpha = self.alpha.to(device=self.device)

        # create embeddings
        h = self.embedding(x)

        # pass through transformer layers
        for idx, layer in enumerate(self.gta3_layers):
            if self.per_layer_alpha: 
                h = layer.forward(h, A, self.alpha[idx])
            else:
                h = layer.forward(h, A, self.alpha)

        # combine resulting node embeddings
        h = torch.mean(h, dim=-2) # TODO: using mean for now

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        x, A, y_true = batch

        # forward pass
        y_pred = self.forward_step(x, A)
        
        # compute loss
        if self.train_alpha:
            train_loss = self.train_loss_func(y_pred, y_true, self.alpha, self.alpha_weight) # NOTE: might not yet work for per head alpha
        else:
            train_loss = self.train_loss_func(y_pred, y_true)

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha/alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=1)
        else:
            self.log("alpha/alpha_0", self.alpha, on_epoch=False, on_step=True, batch_size=1)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=1)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        x, A, y_true = batch

        # forward pass
        y_pred = self.forward_step(x, A)

        # compute loss
        valid_loss = self.valid_loss_func(y_pred, y_true)

        # log loss
        self.log("valid_loss", valid_loss, on_epoch=True, on_step=False, batch_size=1)

        return valid_loss
    