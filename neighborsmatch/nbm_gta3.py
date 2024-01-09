import torch
import torch.nn as nn
import os.path as osp
from torchmetrics.classification import MulticlassAccuracy
from dgl import save_graphs, load_graphs
from dgl.data.utils import save_info, load_info

from neighborsmatch.tree_dataset import TreeDataset

from gta3.model import GTA3BaseModel
from gta3.dataloader import GTA3BaseDataset, transform_to_graph_list


class GTA3_NBM_Dataset(GTA3BaseDataset):

    def __init__(self, mode, phi_func, pos_enc, tree_depth, batch_size=10, force_reload=False, force_regenerate=False, generator_seed=None, pos_enc_dim=8):
        self.mode = mode
        self.depth = tree_depth
        self.generator_seed = generator_seed
        self.raw_path = osp.join(".", ".dgl", f"nbmraw_{self.depth}")
        self.force_regenerate = force_regenerate

        super().__init__('nbm', mode, phi_func, pos_enc, batch_size=batch_size, force_reload=force_reload, pos_enc_dim=pos_enc_dim, path_suffix=f'_{self.depth}')


    def _generate_data(self):

        # generate the data
        generator = TreeDataset(self.depth, self.generator_seed)
        train_data, valid_data, test_data, num_types, num_leaf_nodes = generator.generate_data(train_size=0.8, valid_size=0.5) # -> split into 80% train, 10% valid, 10% test

        # save the data
        save_graphs(osp.join(self.raw_path, "train_data.bin"), [g for g, _ in train_data], {'labels': torch.tensor([[l] for _, l in train_data], dtype=torch.long)})
        save_graphs(osp.join(self.raw_path, "valid_data.bin"), [g for g, _ in valid_data], {'labels': torch.tensor([[l] for _, l in valid_data], dtype=torch.long)})
        save_graphs(osp.join(self.raw_path, "test_data.bin"), [g for g, _ in test_data], {'labels': torch.tensor([[l] for _, l in test_data], dtype=torch.long)})
        save_info(osp.join(self.raw_path, "meta.pkl"), {"num_types": num_types, "num_leaf_nodes": num_leaf_nodes})


    def _load_raw_data(self, data_path, info_path):

        # load raw data
        print(f"Generating the NeighborsMatch {self.mode} data...", end='\r')
        if self.force_regenerate or not osp.exists(self.raw_path):
            self._generate_data()
        print(f"Generating the NeighborsMatch {self.mode} data...Done")

        # load raw data
        print(f"Loading the raw NeighborsMatch {self.mode} data...", end='\r')
        self.graphs, labels_dict = load_graphs(osp.join(self.raw_path, f"{self.mode}_data.bin"))
        self.labels = labels_dict['labels']
        meta_dict = load_info(osp.join(self.raw_path, "meta.pkl"))
        self.num_types = meta_dict['num_types']
        self.num_leaf_nodes = meta_dict['num_leaf_nodes']
        print(f"Loading the raw NeighborsMatch {self.mode} data.......Done")

        # preprocess data
        print(f"Preprocessing the {self.mode} data...", end='\r')
        self._preprocess_data()
        print(f"Preprocessing the {self.mode} data..........Done" + ' '*15)

        # get maximum number of nodes
        self.max_nodes = 0
        for g in self.graphs:
            self.max_nodes = max(self.max_nodes, g.num_nodes())

        # store the preprocessed data
        print(f"Caching the preprocessed {self.mode} data...", end='\r')
        save_graphs(data_path, transform_to_graph_list(self.graphs), {"labels": self.labels})
        if self.compute_class_weights: save_info(info_path, {'num_types': self.num_types, 'num_leaf_nodes': self.num_leaf_nodes, 'max_nodes': self.max_nodes, 'class_weights': self.class_weights})
        else:                          save_info(info_path, {'num_types': self.num_types, 'num_leaf_nodes': self.num_leaf_nodes, 'max_nodes': self.max_nodes})
        print(f"Caching the preprocessed {self.mode} data...Done")


    def _load_cached_data(self, data_path, info_path):
        print(f"Loading cached NeighborsMatch {self.mode} data...", end='\r')
        self.graphs, labels_dict = load_graphs(data_path)
        self.labels = labels_dict['labels']
        info = load_info(info_path)
        self.num_types = info['num_types']
        self.num_leaf_nodes = info['num_leaf_nodes']
        self.max_nodes = info['max_nodes']
        self.class_weights = info['class_weights'] if self.compute_class_weights else None
        print(f"Loading cached NeighborsMatch {self.mode} data...Done")


    def _get_label(self, idx):
        return self.labels[idx]


    def get_num_types(self):
        return self.num_types
    

    def get_num_out_types(self):
        return self.num_leaf_nodes
    

    def max_num_nodes(self):
        return self.max_nodes
    


class GTA3_NBM(GTA3BaseModel):
    
    def __init__(self, model_params, train_params):    

        # setup score name and direction for lr scheduler
        self.score_name = "valid_accuracy"
        self.score_direction = "max"
        # initialize the GTA3 base model
        super().__init__(model_params, train_params)

        # need two embeddings: one for the keys and one for the values
        # -> one is already created in the base model
        self.key_embedding = nn.Embedding(model_params['num_out_types'] + 1, model_params['hidden_dim']) # TODO
        self.embedding_contractor = nn.Linear(model_params['hidden_dim'] * 2, model_params['hidden_dim'])

        # final mlp to map the out dimension to a single value
        self.out_mlp = nn.Sequential(nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2), nn.ReLU(), nn.Dropout(), nn.Linear(model_params['out_dim'] * 2, model_params['num_out_types']))
        
        # loss functions
        self.criterion = nn.CrossEntropyLoss()


    def forward_step(self, x, A, pos_enc, lengths):
        """
            Input:
            - x: [B, N, 2]
            - A: [B, N, Emb]
            - lengths: [B]

        """
        self.alpha = self.alpha.to(device=self.device)

        # create embeddings
        x_type, x_key = x[..., 0], x[..., 1]
        h_type = self.embedding(x_type)
        h_key = self.key_embedding(x_key)
        h = torch.concat((h_type, h_key), dim=-1)
        h = self.embedding_contractor(h)
        #h = h_type + h_key

        # add positional embeddings
        if self.use_pos_enc:
            h_pos = self.pos_embedding(pos_enc)
            h = h + h_pos

        # pass through transformer layers
        for idx, layer in enumerate(self.gta3_layers):
            if self.per_layer_alpha:
                h, log_dict = layer.forward(h, A, lengths, self.alpha[idx])
            else:
                h, log_dict = layer.forward(h, A, lengths, self.alpha)
            for key in log_dict:
                val, bs = log_dict[key]
                self.log(key+"_layer"+str(idx), val, on_epoch=True, on_step=False, batch_size=bs)
        # extract embedding of the root node (root is always first node)
        h = h[:,0,:]

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, labels = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, pos_enc, lengths)
        
        # compute loss
        train_loss = self.criterion(preds, labels.squeeze(-1).long())

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha/alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=batch_size)
        else:
            self.log("alpha/alpha_0", self.alpha, on_epoch=False, on_step=True, batch_size=batch_size)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=batch_size)

        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], on_epoch=False, on_step=True, batch_size=batch_size)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, labels = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, pos_enc, lengths)

        # compute accuracy
        total = labels.size(0) if batch_size == 1 else batch_size * labels.size(1)
        preds = torch.argmax(preds, dim=-1)
        accuracy = (preds == labels.squeeze(-1)).sum().float() / total

        # log accuracy
        self.log("valid_accuracy", accuracy, on_epoch=True, on_step=False, batch_size=batch_size)

        return accuracy


    def test_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, labels = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, pos_enc, lengths)

        # compute accuracy
        total = labels.size(0) if batch_size == 1 else batch_size * labels.size(1)
        preds = torch.argmax(preds, dim=-1)
        accuracy = (preds == labels.squeeze(-1)).sum().float() / total

        # log accuracy
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False, batch_size=batch_size)

        return accuracy
    
