import lightning as L
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from dgl.data import ZINCDataset

from gta3.model import GTA3Layer


class GTA3_ZINC_Dataset(Dataset):

    def __init__(self, mode):

        # load raw data
        print(f"Loading ZINC {mode} data...", end='')
        self.raw_data = ZINCDataset(mode=mode, raw_dir='./.dgl/')
        print(f"Done")

        # preprocess data
        print(f"Preprocessing the data...", end='')
        self._preprocess_data()
        print(f"Done")


    def __len__(self):
        return len(self.raw_data)


    def __getitem__(self, idx):
        # return (node features, adjacency matrix, label) tuple
        return self.raw_data[idx][0].ndata['feat'], self.raw_data[idx][0].ndata['adj_mat'], self.raw_data[idx][1]


    def _preprocess_data(self):

        # create the adjacency matrix for each graph
        # TODO: this is horrable I know but it works for now...
        for g, _ in self.raw_data:
            # adj_mat = torch.zeros((g.num_nodes(), g.num_nodes()))
            # u, v = g.edges()
            # for i in range(len(u)):
            #     adj_mat[v[i]][u[i]] = 1
            # g.ndata['adj_mat'] = adj_mat
            g.ndata['adj_mat'] = torch.tensor([-1 for _ in range(g.num_nodes())])

    
    def get_num_types(self):
        return self.raw_data.num_atom_types

        


class GTA3_ZINC(L.LightningModule):
    
    def __init__(self, model_params, train_params):
        super().__init__()

        self.train_params = train_params

        # creates an embedding depending on the node type
        self.embedding = nn.Embedding(model_params['num_types'], model_params['hidden_dim'])
        
        # the main part of the model
        self.gta3_layers = nn.ModuleList(
            [ GTA3Layer(
                in_dim=model_params['hidden_dim'], out_dim=model_params['hidden_dim'], num_heads=model_params['num_heads'], phi=model_params['phi'],
                residual=model_params['residual'], batch_norm=model_params['batch_norm'], attention_bias=model_params['attention_bias']) 
            for _ in range(model_params['num_layers']-1) ]
        )
        self.gta3_layers.append(
            GTA3Layer(
                in_dim=model_params['hidden_dim'], out_dim=model_params['out_dim'], num_heads=model_params['num_heads'], phi=model_params['phi'],
                residual=model_params['residual'], batch_norm=model_params['batch_norm'], attention_bias=model_params['attention_bias'])
        )

        # final mlp to map the out dimension to a single value
        self.out_mlp = nn.Sequential(nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2), nn.ReLU(), nn.Dropout(), nn.Linear(model_params['out_dim'] * 2, 1))
        
        # loss function
        self.loss_func = nn.L1Loss()


    def training_step(self, batch, batch_idx):
        x, A, y_true = batch

        # create embeddings
        h = self.embedding(x)

        # pass through transformer layers
        for layer in self.gta3_layers:
            h = layer.forward(h, A)

        # combine resulting node embeddings
        h = torch.mean(h, dim=-2) # TODO: using mean for now

        # pass through final mlp
        y_pred = self.out_mlp(h)
        
        return self.loss_func(y_pred, y_true)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.train_params['lr'])
        return optimizer
    