import torch
import torch.nn as nn
from dgl import save_graphs, load_graphs
from dgl.data import ZINCDataset
from dgl.data.utils import save_info, load_info

from gta3.model import GTA3BaseModel
from gta3.dataloader import GTA3BaseDataset
from gta3.loss import AlphaRegularizationWrapper

from common.mlp_readout import MLPReadout


class GTA3_ZINC_Dataset(GTA3BaseDataset):

    def __init__(self, mode, phi_func, pos_enc, batch_size=10, force_reload=False, use_caching=True, pos_enc_dim=8):
        self.mode = mode
        super().__init__('zinc', mode, phi_func, pos_enc, batch_size=batch_size, force_reload=force_reload, use_caching=use_caching, pos_enc_dim=pos_enc_dim)


    def _load_raw_data(self, data_path, info_path):

        # load raw data
        print(f"Loading the raw ZINC {self.mode} data...", end='\r')
        raw_data = ZINCDataset(mode=self.mode, raw_dir='./.dgl/')
        self.graphs = raw_data[:][0]
        self.labels = raw_data[:][1]
        self.num_atom_types = raw_data.num_atom_types
        print(f"Loading the raw ZINC {self.mode} data.......Done")

        # preprocess data
        print(f"Preprocessing the {self.mode} data...", end='\r')
        self._preprocess_data()
        print(f"Preprocessing the {self.mode} data..........Done" + ' '*15)

        # store the preprocessed data
        if self.use_caching:
            print(f"Caching the preprocessed {self.mode} data...", end='\r')
            save_graphs(data_path, self.graphs, {'labels': self.labels})
            save_info(info_path, {'num_atom_types': self.num_atom_types})
            print(f"Caching the preprocessed {self.mode} data...Done")


    def _load_cached_data(self, data_path, info_path):
        print(f"Loading cached ZINC {self.mode} data...", end='\r')
        self.graphs, labels_dict = load_graphs(data_path)
        self.labels = labels_dict['labels']
        self.num_atom_types = load_info(info_path)['num_atom_types']
        print(f"Loading cached ZINC {self.mode} data...Done")

    
    def _get_label(self, idx):
        return self.labels[idx].unsqueeze(0)
    

    def get_num_types(self):
        return self.num_atom_types


class GTA3_ZINC(GTA3BaseModel):
    
    def __init__(self, model_params, train_params):

        # set the score direction and name for lr scheduler
        self.score_direction = "min"
        self.score_name = "valid_loss"
        
        # initialize the GTA3 base model
        super().__init__(model_params, train_params)

        # final mlp to map the out dimension to a single value
        self.out_mlp = MLPReadout(model_params['out_dim'], 1)
        
        # loss function
        self.train_loss_func = AlphaRegularizationWrapper(nn.L1Loss(), model_params['alpha_weight'])
        self.valid_loss_func = nn.L1Loss()


    def forward_step(self, x, A, pos_enc, lengths):
        """
            Input:
            - x: [B, N]
            - A: [B, N, Emb]
            - lengths: [B]

        """
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
                
        # combine resulting node embeddings
        h = torch.mean(h, dim=-2) # TODO: using mean for now

        # pass through final mlp
        return self.out_mlp(h)


    def training_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, y_true = batch

        # forward pass
        y_pred = self.forward_step(x, A, pos_enc, lengths)

        # compute loss
        if isinstance(self.alpha, nn.Parameter):
            train_loss = self.train_loss_func(y_pred, y_true, alpha=self.alpha)
        else:
            train_loss = self.train_loss_func(y_pred, y_true, alpha=None)

        # log loss and alpha
        if self.per_layer_alpha:
            for l in range(len(self.gta3_layers)):
                self.log(f"alpha/alpha_{l}", self.alpha[l], on_epoch=False, on_step=True, batch_size=1)
        else:
            self.log("alpha/alpha_0", self.alpha, on_epoch=False, on_step=True, batch_size=1)
        self.log("train_loss", train_loss, on_epoch=True, on_step=False, batch_size=1)

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
        lengths, x, A, pos_enc, y_true = batch

        # forward pass
        y_pred = self.forward_step(x, A, pos_enc, lengths)

        # compute loss
        if isinstance(self.alpha, nn.Parameter):
            valid_train_loss = self.train_loss_func(y_pred, y_true, alpha=self.alpha)
        else:
            valid_train_loss = self.train_loss_func(y_pred, y_true, alpha=None)
        valid_loss = self.valid_loss_func(y_pred, y_true)

        # log loss
        self.log("valid_train_loss", valid_train_loss, on_epoch=True, on_step=False, batch_size=1)
        self.log(self.score_name, valid_loss, on_epoch=True, on_step=False, batch_size=1, prog_bar=True)

        return valid_loss
    

    def test_step(self, batch, batch_idx):
        lengths, x, A, pos_enc, y_true = batch

        # forward pass
        y_pred = self.forward_step(x, A, pos_enc, lengths)

        # compute loss
        valid_loss = self.valid_loss_func(y_pred, y_true)

        # log loss
        self.log("test_loss", valid_loss, on_epoch=True, on_step=False, batch_size=1)

        return valid_loss
    