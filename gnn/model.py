import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from dgl.nn.pytorch.conv import GraphConv, GATConv
import lightning as L


class GNNBaseModel(L.LightningModule):

    def __init__(self, gnn_type, model_params, train_params):
        """
        Graph Neural Network (GNN) base model.
        Used to create the base of a Graph Convolutional Network (GCN) or Graph Attention 
        Network (GAT) model based on the modules from the dgl library.

        Args:
          model_params: dictionary containing the model parameters
          train_params: dictionary containing the training parameters

        """
        super().__init__()

        self.train_params = train_params

        # creates an embedding depending on the node type
        self.embedding = nn.Embedding(model_params['num_in_types'], model_params['hidden_dim'])

        # gnn layers
        # > graph convolutional network
        if gnn_type == 'gcn':
            self.gnn_layers = nn.ModuleList(
                [GraphConv(model_params['hidden_dim'], model_params['hidden_dim'], activation=F.relu) for _ in range(model_params['num_layers'] - 1)]
            )
            self.gnn_layers.append(GraphConv(model_params['hidden_dim'], model_params['out_dim']))
        
        # > graph attention network
        elif gnn_type == 'gat':
            assert model_params['hidden_dim'] % model_params['num_heads'] == 0, f"GNNBaseModel Error: out_dim ({model_params['hidden_dim']}) must be divisible by num_heads ({model_params['num_heads']})!"

            self.gnn_layers = nn.ModuleList(
                [GATConv(model_params['hidden_dim'], model_params['hidden_dim'] // model_params['num_heads'], model_params['num_heads'], activation=F.relu) for _ in range(model_params['num_layers'] - 1)]
            )
            self.gnn_layers.append(GATConv(model_params['hidden_dim'], model_params['out_dim'] // model_params['num_heads'], model_params['num_heads']))
        
        # > unkown gnn -> error
        else:
            raise ValueError(f"GNNBaseModel: Unkown gnn type '{gnn_type}'!")
        
        


    def training_step(self, batch, batch_idx):
        raise NotImplementedError("GCN3BaseModel: Define training_step function!")


    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("GCN3BaseModel: Define validation_step function!")

        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.train_params['lr'])
        return optimizer