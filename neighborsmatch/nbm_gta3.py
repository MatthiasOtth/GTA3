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

    def __init__(self, mode, phi_func, tree_depth, batch_size=10, force_reload=False, force_regenerate=False, generator_seed=None):
        self.mode = mode
        self.depth = tree_depth
        self.generator_seed = generator_seed
        self.raw_path = osp.join(".", ".dgl", f"nbmraw_{self.depth}")
        self.force_regenerate = force_regenerate

        super().__init__('nbm', mode, phi_func, batch_size=batch_size, force_reload=force_reload, path_suffix=f'_{self.depth}')


    def _generate_data(self):

        # generate the data
        generator = TreeDataset(self.depth, self.generator_seed)
        train_data, valid_data, test_data, num_types, num_tree_nodes = generator.generate_data(train_size=0.8, valid_size=0.5) # -> split into 80% train, 10% valid, 10% test

        # save the data
        save_graphs(osp.join(self.raw_path, "train_data.bin"), [g for g, _ in train_data], {'labels': torch.tensor([[l] for _, l in train_data], dtype=torch.long)})
        save_graphs(osp.join(self.raw_path, "valid_data.bin"), [g for g, _ in valid_data], {'labels': torch.tensor([[l] for _, l in valid_data], dtype=torch.long)})
        save_graphs(osp.join(self.raw_path, "test_data.bin"), [g for g, _ in test_data], {'labels': torch.tensor([[l] for _, l in test_data], dtype=torch.long)})
        save_info(osp.join(self.raw_path, "meta.pkl"), {"num_types": num_types, "num_tree_nodes": num_tree_nodes})


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
        self.num_tree_nodes = meta_dict['num_tree_nodes']
        print(f"Loading the raw NeighborsMatch {self.mode} data.......Done")

        # preprocess data
        print(f"Preprocessing the {self.mode} data...", end='\r')
        self._preprocess_data()
        print(f"Preprocessing the {self.mode} data..........Done" + ' '*15)

        # store the preprocessed data
        print(f"Caching the preprocessed {self.mode} data...", end='\r')
        save_graphs(data_path, transform_to_graph_list(self.graphs), {"labels": self.labels})
        if self.compute_class_weights: save_info(info_path, {'num_types': self.num_types, 'num_tree_nodes': self.num_tree_nodes, 'class_weights': self.class_weights})
        else:                          save_info(info_path, {'num_types': self.num_types, 'num_tree_nodes': self.num_tree_nodes})
        print(f"Caching the preprocessed {self.mode} data...Done")


    def _load_cached_data(self, data_path, info_path):
        print(f"Loading cached NeighborsMatch {self.mode} data...", end='\r')
        self.graphs, labels_dict = load_graphs(data_path)
        self.labels = labels_dict['labels']
        info = load_info(info_path)
        self.num_types = info['num_types']
        self.num_tree_nodes = info['num_tree_nodes']
        self.class_weights = info['class_weights'] if self.compute_class_weights else None
        print(f"Loading cached NeighborsMatch {self.mode} data...Done")


    def _get_label(self, idx):
        return self.labels[idx]


    def get_num_types(self):
        return self.num_types
    

    def get_num_out_types(self):
        return self.num_tree_nodes
    


class GTA3_NBM(GTA3BaseModel):
    
    def __init__(self, model_params, train_params):
        
        # initialize the GTA3 base model
        super().__init__(model_params, train_params)

        # final mlp to map the out dimension to a single value
        self.out_mlp = nn.Sequential(nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2), nn.ReLU(), nn.Dropout(), nn.Linear(model_params['out_dim'] * 2, model_params['num_out_types']))
        
        # loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_func = MulticlassAccuracy(model_params['num_out_types'])


    def forward_step(self, x, A, lengths):
        """
            Input:
            - x: [B, N, 2]
            - A: [B, N, Emb]
            - lengths: [B]

        """
        self.alpha = self.alpha.to(device=self.device)

        # create embeddings
        h = self.embedding(x)

        # pass through transformer layers
        for idx, layer in enumerate(self.gta3_layers):
            if self.per_layer_alpha: 
                h = layer.forward(h, A, lengths, self.alpha[idx])
            else:
                h = layer.forward(h, A, lengths, self.alpha)

        # extract embedding of the root node (root is always first node)
        h = h[:,0,:]

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        lengths, x, A, labels = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, lengths)

        # compute loss
        train_loss = self.criterion(preds, labels.squeeze(-1).long())

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha/alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=batch_size)
        else:
            self.log("alpha/alpha_0", self.alpha, on_epoch=False, on_step=True, batch_size=batch_size)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=batch_size)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        lengths, x, A, labels = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, lengths)

        # compute accuracy
        valid_loss = self.accuracy_func(preds, labels.squeeze(-1).long())

        # log accuracy
        self.log("valid_accuracy", valid_loss, on_epoch=True, on_step=False, batch_size=batch_size)

        return valid_loss
    