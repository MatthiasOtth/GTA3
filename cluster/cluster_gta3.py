import torch
import torch.nn as nn
import einops
from torchmetrics.classification import MulticlassAccuracy
from dgl import save_graphs, load_graphs
from dgl.data import CLUSTERDataset
from dgl.data.utils import save_info, load_info

from gta3.model import GTA3BaseModel
from gta3.dataloader import GTA3BaseDataset, transform_to_graph_list
from gta3.loss import AlphaRegularizationWrapper
from common.mlp_readout import MLPReadout


class GTA3_CLUSTER_Dataset(GTA3BaseDataset):

    def __init__(self, mode, phi_func, pos_enc, batch_size=10, force_reload=False, use_caching=True, pos_enc_dim=8):
        self.mode = mode
        print(f"batch_size = {batch_size}")
        super().__init__('cluster', mode, phi_func, pos_enc, batch_size=batch_size, force_reload=force_reload, use_caching=use_caching, pos_enc_dim=pos_enc_dim, compute_class_weights=True)


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
        if self.use_caching:
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

        # init score name and direction for lr scheduler
        self.score_name = 'valid_accuracy'
        self.score_direction = 'max'
        
        # initialize the GTA3 base model
        super().__init__(model_params, train_params)

        # final mlp to map the out dimension to a single value
        self.out_mlp = MLPReadout(model_params['out_dim'], model_params['num_out_types'])
        
        # loss functions
        self.criterion = AlphaRegularizationWrapper(nn.CrossEntropyLoss(), model_params['alpha_weight'])


    def forward_step(self, x, A, pos_enc, lengths):
        """
            Input:
            - x: [B, N]
            - A: [B, N, Emb]
            - lengths: [B]

        """
        self.alpha = self.alpha.to(device=self.device)

        # create embeddings
        h = self.embedding(x)

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
                self.log(key, val, on_epoch=True, on_step=False, batch_size=bs)
        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, labels, class_weights = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, pos_enc, lengths)
        
        # compute loss
        if batch_size > 1:
            preds = einops.rearrange(preds, 'b n d -> (b n) d')
            labels = einops.rearrange(labels, 'b l -> (b l)')
        self.criterion.weight = class_weights
        
        if isinstance(self.alpha, nn.Parameter):
            train_loss = self.criterion(preds, labels.long(), self.alpha)
        else:
            train_loss = self.criterion(preds, labels.long())

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha/alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=batch_size)
        else:
            self.log("alpha/alpha_0", self.alpha, on_epoch=False, on_step=True, batch_size=batch_size)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=batch_size)

        for io, opt in enumerate(self.trainer.optimizers):
            prefix = f"optim_{io}_" if len(self.trainer.optimizers) > 1 else ""
            for i, param_group in enumerate(opt.param_groups):
                if 'lr' in param_group:
                    self.log(
                        "lr/" + prefix + f"group_{i}_lr", param_group['lr'], 
                        on_epoch=True, on_step=False, batch_size=1
                    )

        return train_loss
    

    def validation_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, labels, _ = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, pos_enc, lengths)

        # compute accuracy
        total = labels.size(0) if batch_size == 1 else batch_size * labels.size(1)
        preds = torch.argmax(preds, dim=-1)
        accuracy = (preds == labels).sum().float() / total

        # log accuracy
        self.log(self.score_name, accuracy, on_epoch=True, on_step=False, batch_size=total, prog_bar=True)

        return accuracy


    def test_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, labels, _ = batch
        batch_size = 1 if len(labels.shape) == 1 else labels.size(0)

        # forward pass
        preds = self.forward_step(x, A, pos_enc, lengths)

        # compute accuracy
        total = labels.size(0) if batch_size == 1 else batch_size * labels.size(1)
        preds = torch.argmax(preds, dim=-1)
        accuracy = (preds == labels).sum().float() / total

        # log accuracy
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False, batch_size=total)

        return accuracy
    