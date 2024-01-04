import torch.nn as nn
import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data import ZINCDataset
from dgl import add_self_loop

from gnn.model import GNNBaseModel
from common.mlp_readout import MLPReadout


class GNN_ZINC_DataLoader(GraphDataLoader):

    def __init__(self, mode, batch_size=10):

        # load the cluster data
        print(f"Loading the {mode} ZINC data...", end='\r')
        self.dataset = ZINCDataset(mode=mode, raw_dir='./.dgl/')
        self.num_types = self.dataset.num_atom_types
        print(f"Loading the {mode} ZINC data...Done")

        print(f"Adding self loops to the data...", end='\r')
        self.dataset = list(map(lambda d: (add_self_loop(d[0]), d[1]), self.dataset))
        print(f"Adding self loops to the data...Done")

        # setup the data loader
        super().__init__(self.dataset, batch_size=batch_size, drop_last=False)


    def get_num_in_types(self):
        return self.num_types



class GNN_ZINC(GNNBaseModel):

    def __init__(self, gnn_type, model_params, train_params):

        # initialize the GNN base model
        super().__init__(gnn_type, model_params, train_params)

        # final readout mlp
        self.out_mlp = MLPReadout(model_params['out_dim'], 1)

        # loss function
        self.loss_func = nn.L1Loss()

        # gat flag -> we need to reshape the output of the conv layer
        self.model_gat = False
        if gnn_type == 'gat':
            self.model_gat = True
            print("self.model_gat = True")


    def forward_step(self, g):
        x = g.ndata['feat']

        # create embeddings
        h = self.embedding(x)
        
        # pass through GNN layers
        for layer in self.gnn_layers:
            h = layer(g, h)
            if self.model_gat:
                h = h.reshape(h.size(0), -1)
            
        # combine resulting node embeddings
        g.ndata['h'] = h
        h = dgl.readout_nodes(g, 'h', op='mean')

        # pass through readout mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        g, y_true = batch
        batch_size = len(g.batch_num_nodes())

        # forward pass
        y_pred = self.forward_step(g).squeeze(-1)

        # compute loss 
        train_loss = self.loss_func(y_pred, y_true)

        # log loss
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=batch_size)

        return train_loss


    def validation_step(self, batch, batch_idx):
        g, y_true = batch
        batch_size = len(g.batch_num_nodes())

        # forward pass
        y_pred = self.forward_step(g).squeeze(-1)

        # compute accuracy
        valid_loss = self.loss_func(y_pred, y_true)

        # log accuracy
        self.log("valid_loss", valid_loss, on_epoch=True, on_step=False, batch_size=batch_size)

        return valid_loss