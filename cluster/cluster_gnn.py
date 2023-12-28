import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
from dgl.data import CLUSTERDataset
from dgl import add_self_loop

from gnn.model import GNNBaseModel


class GNN_CLUSTER_DataLoader(GraphDataLoader):

    def __init__(self, mode, batch_size=10):

        # load the cluster data
        print(f"Loading the {mode} CLUSTER data...", end='\r')
        self.dataset = CLUSTERDataset(mode=mode, raw_dir='./.dgl/')
        print(f"Loading the {mode} CLUSTER data...Done")

        # setup the data loader
        super().__init__(self.dataset, batch_size=batch_size, drop_last=False)


    def get_num_in_types(self):
        return self.dataset.num_classes + 1
    

    def get_num_out_types(self):
        return self.dataset.num_classes



class GNN_CLUSTER(GNNBaseModel):

    def __init__(self, gnn_type, model_params, train_params):

        # initialize the GNN base model
        super().__init__(gnn_type, model_params, train_params)

        # final readout mlp
        self.out_mlp = nn.Sequential(
            nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2), 
            nn.ReLU(), 
            nn.Dropout(), 
            nn.Linear(model_params['out_dim'] * 2, model_params['num_out_types'])
        )

        # loss function
        self.criterion = nn.CrossEntropyLoss()

        # gat flag -> we need to reshape the output of the conv layer
        self.model_gat = False
        if gnn_type == 'gat':
            self.model_gat = True


    def forward_step(self, g):
        x = g.ndata['feat']

        # create embeddings
        h = self.embedding(x)
        
        # pass through GNN layers
        for layer in self.gnn_layers:
            h = layer(g, h)
            if self.model_gat:
                h = h.reshape(h.size(0), -1)

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        g = add_self_loop(batch)
        batch_size = len(g.batch_num_nodes())

        # forward pass
        preds = self.forward_step(g)
        labels = g.ndata['label'].long()

        # compute loss 
        # -> TODO: might want to add class weights
        train_loss = self.criterion(preds, labels)

        # log loss
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=batch_size)

        return train_loss


    def validation_step(self, batch, batch_idx):
        g = add_self_loop(batch)
        batch_size = len(g.batch_num_nodes())

        # forward pass
        preds = self.forward_step(g)
        labels = g.ndata['label']

        # compute accuracy
        total = labels.size(0)
        preds = torch.argmax(preds, dim=-1)
        accuracy = (preds == labels).sum().float() / total

        # log accuracy
        self.log("valid_accuracy", accuracy, on_epoch=True, on_step=False, batch_size=batch_size)

        return accuracy