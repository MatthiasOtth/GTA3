import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
import networkx as nx
from dgl.data import ZINCDataset

from gta3.model import GTA3Layer
from gta3.loss import L1Loss_L1Alpha, L1Loss_L2Alpha


class GTA3_ZINC_Dataset(Dataset):

    def __init__(self, mode, phi_func):

        # load raw data
        print(f"Loading ZINC {mode} data...", end='\r')
        self.raw_data = ZINCDataset(mode=mode, raw_dir='./.dgl/')
        print(f"Loading ZINC {mode} data...Done")

        # determine necessary precomputation steps
        self.use_shortest_path = False
        self.use_adj_matrix = False
        if phi_func == 'test':
            self.use_adj_matrix = True
        elif phi_func == 'inverse_hops':
            self.use_shortest_path = True

        # preprocess data
        print(f"Preprocessing the data...", end='\r')
        self._preprocess_data()
        print(f"Preprocessing the data....Done")


    def __len__(self):
        return len(self.raw_data)


    def __getitem__(self, idx):
        # return (node features, adjacency matrix, label) tuple
        if self.use_adj_matrix:
            return self.raw_data[idx][0].ndata['feat'], self.raw_data[idx][0].ndata['adj_mat'], self.raw_data[idx][1].unsqueeze(0)
        elif self.use_shortest_path:
            return self.raw_data[idx][0].ndata['feat'], self.raw_data[idx][0].ndata['short_dist_mat'], self.raw_data[idx][1].unsqueeze(0)
        else:
            return self.raw_data[idx][0].ndata['feat'], None, self.raw_data[idx][1].unsqueeze(0)


    def _preprocess_data(self):

        if self.use_adj_matrix or self.use_shortest_path:
            for g, _ in self.raw_data:
                
                # create the adjacency matrix for each graph
                # TODO: this is horrable I know but it works for now...
                adj_mat = torch.zeros((g.num_nodes(), g.num_nodes()))
                u, v = g.edges()
                for i in range(len(u)):
                    adj_mat[u[i]][v[i]] = 1
                # adj_mat = F.softmax(adj_mat, dim=1) # TODO: tryout
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
        return self.raw_data.num_atom_types

        
class GTA3_ZINC(L.LightningModule):
    
    def __init__(self, model_params, train_params):
        super().__init__()

        self.train_params = train_params

        # initialize the alpha value
        if model_params['alpha'] == 'fixed':
            self.per_layer_alpha = False
            self.alpha = float(model_params['alpha_init'])
        elif model_params['alpha'] == 'per_model': # TODO: test this...
            self.per_layer_alpha = False
            self.alpha = torch.nn.Parameter(torch.tensor([model_params['alpha_init']], dtype=torch.float))
        elif model_params['alpha'] == 'per_layer': # TODO: test this...
            self.per_layer_alpha = True
            self.alpha = torch.nn.Parameter(torch.tensor([model_params['alpha_init'] for _ in range(model_params['num_layers'])], dtype=torch.float))
        elif model_params['alpha'] == 'per_head': # TODO: test this...
            self.per_layer_alpha = True
            # TODO: implement
            raise NotImplementedError("alpha per head is not yet implemented...")
        else:
            raise ValueError("Invalid alpha model parameter!", model_params['alpha'])

        # creates an embedding depending on the node type
        self.embedding = nn.Embedding(model_params['num_types'], model_params['hidden_dim'])
        
        # the main part of the model
        self.gta3_layers = nn.ModuleList(
            [ GTA3Layer(
                in_dim=model_params['hidden_dim'], out_dim=model_params['hidden_dim'], num_heads=model_params['num_heads'], phi=model_params['phi'],
                residual=model_params['residual'], batch_norm=model_params['batch_norm'], layer_norm=model_params['layer_norm'], 
                attention_bias=model_params['attention_bias']) 
            for _ in range(model_params['num_layers']-1) ]
        )
        self.gta3_layers.append(
            GTA3Layer(
                in_dim=model_params['hidden_dim'], out_dim=model_params['out_dim'], num_heads=model_params['num_heads'], phi=model_params['phi'],
                residual=model_params['residual'], batch_norm=model_params['batch_norm'], layer_norm=model_params['layer_norm'],
                attention_bias=model_params['attention_bias'])
        )

        # final mlp to map the out dimension to a single value
        self.out_mlp = nn.Sequential(
            nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2),
            nn.ReLU(), 
            # nn.Dropout(), 
            nn.Linear(model_params['out_dim'] * 2, model_params['out_dim'] // 2),
            nn.ReLU(), 
            # nn.Dropout(), 
            nn.Linear(model_params['out_dim'] // 2, 1),
        )
        
        # loss functions
        if model_params['alpha'] == 'fixed':
            self.train_alpha = False
            self.train_loss = nn.L1Loss()
        else:
            self.train_alpha = True
            self.train_loss = L1Loss_L1Alpha
            self.alpha_weight = model_params['lambda']
        self.valid_loss = nn.L1Loss()


    def forward_step(self, x, A):

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

        # perform forward pass
        y_pred = self.forward_step(x, A)
        
        # compute loss
        if self.train_alpha:
            train_loss = self.train_loss(y_pred, y_true, self.alpha, self.alpha_weight) # NOTE: might not yet work for per head alpha
        else:
            train_loss = self.train_loss(y_pred, y_true)

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=1)
        else:
            self.log("alpha", self.alpha, on_epoch=False, on_step=True, batch_size=1)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=1)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        x, A, y_true = batch

        # perform forward pass
        y_pred = self.forward_step(x, A)

        # compute loss
        valid_loss = self.valid_loss(y_pred, y_true)

        # log loss
        self.log("valid_loss", valid_loss, on_epoch=True, on_step=False, batch_size=1)

        return valid_loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.train_params['lr'])
        return optimizer
    