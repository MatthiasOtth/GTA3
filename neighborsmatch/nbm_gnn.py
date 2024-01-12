import torch
import torch.nn as nn
import os.path as osp
from dgl.dataloading import GraphDataLoader
from dgl import add_self_loop, save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

from neighborsmatch.tree_dataset import TreeDataset
from gnn.model import GNNBaseModel


class GNN_NBM_DataLoader(GraphDataLoader):

    def __init__(self, mode, tree_depth, batch_size=10, force_regenerate=False, generator_seed=None):
        self.depth = tree_depth
        self.generator_seed = generator_seed
        self.raw_path = osp.join(".", ".dgl", f"nbmraw_{self.depth}")

        # generate data
        print(f"Generating the NeighborsMatch {mode} data...", end='\r')
        if force_regenerate or not osp.exists(self.raw_path):
            self._generate_data()
        print(f"Generating the NeighborsMatch {mode} data...Done")

        # load the cluster data
        print(f"Loading the {mode} NeighborsMatch data...", end='\r')
        self.graphs, labels_dict = load_graphs(osp.join(self.raw_path, f"{mode}_data.bin"))
        self.labels = labels_dict['labels']
        meta_dict = load_info(osp.join(self.raw_path, "meta.pkl"))
        self.num_types = meta_dict['num_types']
        self.num_leaf_nodes = meta_dict['num_leaf_nodes']
        print(f"Loading the {mode} NeighborsMatch data...Done")

        print(f"Adding self loops to the data...", end='\r')
        self.graphs = list(map(add_self_loop, self.graphs))
        print(f"Adding self loops to the data...Done")

        self.max_nodes = 0
        for g in self.graphs:
            self.max_nodes = max(self.max_nodes, g.num_nodes())

        # setup the data loader
        assert len(self.labels) == len(self.graphs), "AssertionError: Should have the same number of labels and graphs!"
        dataset = [(self.graphs[i], self.labels[i]) for i in range(len(self.labels))]
        super().__init__(dataset, batch_size=batch_size, drop_last=False)


    def _generate_data(self): # TODO: same as in gta3 dataset -> might want to move into tree_dataset.py

        # generate the data
        generator = TreeDataset(self.depth, self.generator_seed)
        train_data, valid_data, test_data, num_types, num_leaf_nodes = generator.generate_data(train_size=0.8, valid_size=0.5) # -> split into 80% train, 10% valid, 10% test

        # save the data
        save_graphs(osp.join(self.raw_path, "train_data.bin"), [g for g, _ in train_data], {'labels': torch.tensor([[l] for _, l in train_data], dtype=torch.long)})
        save_graphs(osp.join(self.raw_path, "valid_data.bin"), [g for g, _ in valid_data], {'labels': torch.tensor([[l] for _, l in valid_data], dtype=torch.long)})
        save_graphs(osp.join(self.raw_path, "test_data.bin"), [g for g, _ in test_data], {'labels': torch.tensor([[l] for _, l in test_data], dtype=torch.long)})
        save_info(osp.join(self.raw_path, "meta.pkl"), {"num_types": num_types, "num_leaf_nodes": num_leaf_nodes})


    def get_num_in_types(self):
        return self.num_types
    

    def get_num_out_types(self):
        return self.num_leaf_nodes


    def max_num_nodes(self):
        return self.max_nodes
    


class GNN_NBM(GNNBaseModel):
    
    def __init__(self, gnn_type, model_params, train_params):

        # initialize the GNN base model
        super().__init__(gnn_type, model_params, train_params)

        # need two embeddings: one for the keys and one for the values
        # -> one is already created in the base model
        self.key_embedding = nn.Embedding(model_params['num_out_types'] + 1, model_params['hidden_dim']) # TODO
        self.embedding_contractor = nn.Linear(model_params['hidden_dim'] * 2, model_params['hidden_dim'])

        # set the score direction and name for lr scheduler
        self.score_direction = 'min'
        self.score_name = 'train_loss'
        
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
        x_type, x_key = x[..., 0], x[..., 1]
        h_type = self.embedding(x_type)
        h_key = self.key_embedding(x_key)
        h = torch.concat((h_type, h_key), dim=-1)
        h = self.embedding_contractor(h)
        #h = h_type + h_key
        
        # pass through GNN layers
        for i, layer in enumerate(self.gnn_layers):
            h_prev = h

            # > gnn layer
            h = layer(g, h)
            if self.model_gat:
                h = h.reshape(h.size(0), -1)
            
            # > residual connection
            if self.use_residual:
                h = h + h_prev
            
            # > layer norm
            if self.use_layer_norm:
                h = self.layer_norm[i](h)

        # extract embedding of the root node (root is always first node)
        root_indexes = g.batch_num_nodes()
        root_indexes = torch.cumsum(root_indexes, dim=0) - root_indexes[0]
        h = h[root_indexes]

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        g, labels = batch
        batch_size = len(g.batch_num_nodes())

        # forward pass
        preds = self.forward_step(g)

        # compute loss 
        # -> TODO: might want to add class weights
        train_loss = self.criterion(preds, labels.squeeze(-1))

        # log loss
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=batch_size)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=False, on_step=True, batch_size=batch_size)

        return train_loss


    def validation_step(self, batch, batch_idx):
        g, labels = batch
        batch_size = len(g.batch_num_nodes())

        # forward pass
        preds = self.forward_step(g)

        # compute accuracy
        total = labels.size(0)
        preds = torch.argmax(preds, dim=-1)
        accuracy = (preds == labels.squeeze(-1)).sum().float() / total

        # log accuracy
        self.log("valid_accuracy", accuracy, on_epoch=True, on_step=False, batch_size=batch_size)

        return accuracy
    
    def test_step(self, batch, batch_idx):
        g, labels = batch
        batch_size = len(g.batch_num_nodes())

        # forward pass
        preds = self.forward_step(g)

        # compute accuracy
        total = labels.size(0)
        preds = torch.argmax(preds, dim=-1)
        accuracy = (preds == labels.squeeze(-1)).sum().float() / total

        # log accuracy
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False, batch_size=batch_size)

        return accuracy