import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
from dgl import save_graphs, load_graphs
from dgl.data import CLUSTERDataset
from dgl.data.utils import save_info, load_info

from gta3.model import GTA3BaseModel
from gta3.dataloader import GTA3BaseDataset


class GTA3_CLUSTER_Dataset(GTA3BaseDataset):

    def __init__(self, mode, phi_func, force_reload=False):
        self.mode = mode
        super().__init__('cluster', mode, phi_func, force_reload=force_reload, compute_class_weights=True)


    def __getitem__(self, idx):
        # return (node features, adjacency matrix, label) tuple
        if self.use_adj_matrix:
            return (
                self.graphs[idx].ndata['feat'],
                self.graphs[idx].ndata['adj_mat'],
                self.graphs[idx].ndata['label'],
                self.graphs[idx].ndata['class_weights'])
        elif self.use_shortest_path:
            return (
                self.graphs[idx].ndata['feat'],
                self.graphs[idx].ndata['short_dist_mat'],
                self.graphs[idx].ndata['label'],
                self.graphs[idx].ndata['class_weights'])
        else:
            return (
                self.graphs[idx].ndata['feat'],
                None,
                self.graphs[idx]['label'],
                self.graphs[idx].ndata['class_weights'])


    def _load_raw_data(self, data_path, info_path):

        # load raw data
        print(f"Loading the raw CLUSTER {self.mode} data...", end='\r')
        self.graphs = CLUSTERDataset(mode=self.mode, raw_dir='./.dgl/')
        self.num_classes = self.graphs.num_classes
        print(f"Loading the raw CLUSTER {self.mode} data.......Done")

        # preprocess data
        print(f"Preprocessing the {self.mode} data...", end='\r')
        self._preprocess_data()
        print(f"Preprocessing the {self.mode} data..........Done")

        # store the preprocessed data
        print(f"Caching the preprocessed {self.mode} data...", end='\r')
        save_graphs(data_path, self.graphs)
        save_info(info_path, {'num_classes': self.num_classes})
        print(f"Caching the preprocessed {self.mode} data...Done")


    def _load_cached_data(self, data_path, info_path):
        print(f"Loading cached CLUSTER {self.mode} data...", end='\r')
        self.graphs = load_graphs(data_path)
        self.num_classes = load_info(info_path)['num_classes']
        print(f"Loading cached CLUSTER {self.mode} data...Done")


    def get_num_classes(self):
        return self.num_classes
        


class GTA3_CLUSTER(GTA3BaseModel):
    
    def __init__(self, model_params, train_params):
        
        # initialize the GTA3 base model
        super().__init__(model_params, train_params)

        # final mlp to map the out dimension to a single value
        self.out_mlp = nn.Sequential(nn.Linear(model_params['out_dim'], model_params['out_dim'] * 2), nn.ReLU(), nn.Dropout(), nn.Linear(model_params['out_dim'] * 2, model_params['num_classes']))
        
        # loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_func = MulticlassAccuracy(model_params['num_classes'])


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

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        x, A, labels, class_weights = batch

        # forward pass
        preds = self.forward_step(x, A)
        
        # compute loss
        self.criterion.weight = class_weights
        train_loss = self.train_loss_func(preds, labels)

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha/alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=1)
        else:
            self.log("alpha/alpha_0", self.alpha, on_epoch=False, on_step=True, batch_size=1)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=1)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        x, A, labels, _ = batch

        # forward pass
        preds = self.forward_step(x, A)

        # compute accuracy
        valid_loss = self.accuracy_func(preds, labels)

        # log accuracy
        self.log("valid_accuracy", valid_loss, on_epoch=True, on_step=False, batch_size=1)

        return valid_loss
    