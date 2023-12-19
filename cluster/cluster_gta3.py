import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
from dgl import save_graphs, load_graphs
from dgl.data import CLUSTERDataset
from dgl.data.utils import save_info, load_info

from gta3.model import GTA3BaseModel
from gta3.dataloader import GTA3BaseDataset, transform_to_graph_list


class GTA3_CLUSTER_Dataset(GTA3BaseDataset):

    def __init__(self, mode, phi_func, batch_size=10, force_reload=False):
        self.mode = mode
        super().__init__('cluster', mode, phi_func, batch_size=batch_size, force_reload=force_reload, compute_class_weights=True)


    def _load_raw_data(self, data_path, info_path):

        # load raw data
        print(f"Loading the raw CLUSTER {self.mode} data...", end='\r')
        self.graphs = CLUSTERDataset(mode=self.mode, raw_dir='./.dgl/')
        self.num_classes = self.graphs.num_classes
        print(f"Loading the raw CLUSTER {self.mode} data.......Done")

        # preprocess data
        print(f"Preprocessing the {self.mode} data...", end='\r')
        self._preprocess_data()
        print(f"Preprocessing the {self.mode} data..........Done" + ' '*15)

        # store the preprocessed data
        print(f"Caching the preprocessed {self.mode} data...", end='\r')
        save_graphs(data_path, transform_to_graph_list(self.graphs))
        if self.compute_class_weights: save_info(info_path, {'num_classes': self.num_classes, 'class_weights': self.class_weights})
        else:                          save_info(info_path, {'num_classes': self.num_classes})
        print(f"Caching the preprocessed {self.mode} data...Done")


    def _load_cached_data(self, data_path, info_path):
        print(f"Loading cached CLUSTER {self.mode} data...", end='\r')
        self.graphs = load_graphs(data_path)[0]
        info = load_info(info_path)
        self.num_classes = info['num_classes']
        self.class_weights = info['class_weights'] if self.compute_class_weights else None
        print(f"Loading cached CLUSTER {self.mode} data...Done")


    def _get_label(self, idx):
        return self.graphs[idx].ndata['label']


    def get_num_types(self):
        return self.num_classes + 1
    

    def get_num_out_types(self):
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


    def forward_step(self, x, A, lengths):
        """
            Input:
            - x: [B, N]
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

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        lengths, x, A, labels, class_weights = batch

        # forward pass
        preds = self.forward_step(x, A, lengths)
        
        # compute loss
        self.criterion.weight = class_weights
        train_loss = self.criterion(preds, labels.long())

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha/alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=1)
        else:
            self.log("alpha/alpha_0", self.alpha, on_epoch=False, on_step=True, batch_size=1)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=1)

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        lengths, x, A, labels, _ = batch

        # forward pass
        preds = self.forward_step(x, A, lengths)

        # compute accuracy
        valid_loss = self.accuracy_func(preds, labels)

        # log accuracy
        self.log("valid_accuracy", valid_loss, on_epoch=True, on_step=False, batch_size=1)

        return valid_loss
    