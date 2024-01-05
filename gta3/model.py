import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import lightning as L
from torch import optim


def phi_no_weighting(a, A, alpha):
    return a

def phi_simple_adj_weighting(a, A, alpha):
    new_a = a * (1-alpha) + A * alpha
    return new_a

def phi_local(a, A, alpha):
    """ assumes A is adjacency matrix """
    new_a = torch.where(A==1, a, torch.zeros_like(a))
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a

def phi_alpha_pow_dist(a, A, alpha):
    alpha = torch.clamp(alpha, min=0) #, max=1)
    new_a = torch.pow(alpha + 1e-10, A) * a
    new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a

def phi_alpha_pow_dist_exp(a, A, alpha):
    """ a * (e^alpha)^A where A is (transposed) shortest path matrix """
    new_a = a * torch.exp(A * alpha)
    new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a

def alpha_pow_dist_sigmoid(a, A, alpha):
    """ a * sigmoid(alpha)^A where A is (transposed) shortest path matrix """
    new_a = a * torch.sigmoid(A * alpha)
    new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    return new_a

def phi_alpha_pow_dist_sigmoid(a, A, alpha):
    """ a * sigmoid(alpha)^A where A is (transposed) shortest path matrix """
    new_a = alpha_pow_dist_sigmoid(a, A, alpha)
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a

def phi_alpha_pow_dist_sigmoid_softmax(a, A, alpha):
    """ a * sigmoid(alpha)^A where A is (transposed) shortest path matrix """
    new_a = alpha_pow_dist_sigmoid(a, A, alpha)
    new_a = F.softmax(new_a, dim=-1)
    return new_a

def phi_inverse_hops(a, A, alpha):
    """ a * A^{-1/alpha} """
    alpha = torch.clamp(alpha, min=1e-8)
    new_a = a * torch.pow(A, -1/alpha)
    new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a

def phi_inverse_hops_exp(a, A, alpha):
    """ a * A^{-1/exp(alpha)} """
    new_a = a * torch.pow(A, -1/torch.exp(alpha))
    new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a

def phi_poisson_exp(a, A, alpha):
    """ a * poisson(A; alpha) """
    # log prob from torch.distributions.poisson: value.xlogy(rate) - rate - (value + 1).lgamma()
    rate_log = alpha  # we learn exp(rate) bc it is always positive
    rate = torch.exp(rate_log)
    value = A
    log_prob = value * rate_log - rate - (value + 1).lgamma()
    new_a = a * torch.exp(log_prob)
    new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a


class AdjacencyAwareMultiHeadAttention(nn.Module):

    def __init__(self, in_dim, out_dim, phi, num_heads=8, bias=True):
        """
        Adjacency aware multi head attention layer.

        Args:
          in_dim:    the dimension of the input vectors
          out_dim:   the dimension of the output vectors
          phi:       the weighting function to be used
          num_heads: the number of attention heads to be used (default: 8)
          bias:      whether the attention matrices (Q, K, V) use a bias or not (default: True)
        """
        super().__init__()

        self.out_dim = out_dim
        self.sqrt_out_dim = torch.sqrt(torch.tensor(out_dim))

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=bias)

        self.softmax = nn.Softmax(dim=-1)
        self.phi = phi


    def forward(self, h, A, lengths, alpha):
        """
        Computes an adjacency aware multi head attention forward pass.

        Args:
          h:     node embeddings [batch, nodes, in_dim] or [nodes, in_dim]
          A:     matrix provided to the weighting function [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]
          lengths: [batch,]
          alpha: value used by the weighting function

        """
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
            A = A.unsqueeze(0)

        assert len(h.shape) == 3, f"AdjacencyAwareMultiHeadAttention Error: Got invalid shape for h {h.shape}!"

        # apply the query, key and value matrices
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        Q_h = einops.rearrange(Q_h, 'b n (k d) -> b k n d', d=self.out_dim)
        K_h = einops.rearrange(K_h, 'b n (k d) -> b k n d', d=self.out_dim)
        V_h = einops.rearrange(V_h, 'b n (k d) -> b k n d', d=self.out_dim)

        # compute the dot products
        attention = torch.matmul(Q_h, K_h.transpose(-1,-2))  # [batch, heads, nodes, nodes]

        #TODO: Optimize generation of mask
        mask = torch.full((attention.shape[0], attention.shape[2]), False, device=attention.device)
        for i in range(lengths.shape[0]):
            mask[i,lengths[i]:] = True
        mask.unsqueeze_(1)
        mask.unsqueeze_(2)
        #Mask: [batch, 1, 1, nodes]
        attention = attention.masked_fill(mask, float('-inf'))
        # apply scaling and softmax to attention
        attention = self.softmax(attention / self.sqrt_out_dim)

        mask = mask.moveaxis(-2,-1)
        # Mask: [batch, 1, nodes, 1]
        attention = attention.masked_fill(mask, float(0))
        # reweight using the adjacency information
        attention = attention.moveaxis(1,0)
        log_dict = {}
        A_t = A.transpose(-1,-2)
        log_dict["attention/attention_pre_transform_d1"]  = attention[:,A_t==1].mean(), attention[:,A_t==1].reshape(-1).shape[0]
        log_dict["attention/attention_pre_transform_d2+"] = attention[:,A_t >1].mean(), attention[:,A_t >1].reshape(-1).shape[0]

        attention = self.phi(attention, A_t, alpha)

        log_dict["attention/attention_post_transform_d1"]  = attention[:,A_t==1].mean(), attention[:,A_t==1].reshape(-1).shape[0]
        log_dict["attention/attention_post_transform_d2+"] = attention[:,A_t >1].mean(), attention[:,A_t >1].reshape(-1).shape[0]
        attention = attention.moveaxis(0,1)
        
        # sum value tensors scaled by the attention weights
        h_heads = torch.matmul(attention, V_h)
        return h_heads, log_dict


class GTA3Layer(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
        phi,
        num_heads=8,
        residual=True,
        batch_norm=False,
        layer_norm=True,
        attention_bias=True,
    ):
        """
        Graph Transformer with Adjacency Aware Attention (GTA3) layer.

        Args:
          in_dim:         the dimension of the input vectors
          out_dim:        the dimension of the output vectors
          phi:            the weighting function to be used
          num_heads:      the number of attention heads to be used (default: 8)
          residual:       whether to use residual connections (default: True) (note that if in_dim != out_dim the heads will not have a residual connection)
          batch_norm:     whether to use batch normalization (default: False)
          layer_norm:     whether to use layer normalization (default: True)
          attention_bias: whether the attention matrices (Q, K, V) use a bias or not (default: True)
        """
        super().__init__()
        assert out_dim%num_heads == 0, f"GTA3 Error: out_dim ({out_dim}) must be divisible by num_heads ({num_heads})!"
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.residual_heads = residual and in_dim == out_dim
        self.residual_ffn = residual

        if phi == 'none':
            self.phi = phi_no_weighting
        elif phi == 'local':
            self.phi = phi_local
        elif phi == 'test':
            self.phi = phi_simple_adj_weighting
        elif phi == 'inverse_hops':
            self.phi = phi_inverse_hops
        elif phi == 'inverse_hops_exp':
            self.phi = phi_inverse_hops_exp
        elif phi == 'alpha_pow_dist':
            self.phi = phi_alpha_pow_dist
        elif phi == 'alpha_pow_dist_exp':
            self.phi = phi_alpha_pow_dist_exp
        elif phi == 'alpha_pow_dist_sigmoid':
            self.phi = phi_alpha_pow_dist_sigmoid
        elif phi == 'alpha_pow_dist_sigmoid_softmax':
            self.phi = phi_alpha_pow_dist_sigmoid_softmax
        elif phi == 'phi_poisson_exp':
            self.phi = phi_poisson_exp
        else:
            raise NotImplementedError(f"GTA3 Error: Unknown phi function {phi}! Use one of the following: 'none', 'test'")

        self.O = nn.Linear(out_dim, out_dim)
        self.aa_attention = AdjacencyAwareMultiHeadAttention(
            in_dim=in_dim,
            out_dim=out_dim//num_heads,
            phi=self.phi,
            num_heads=num_heads,
            bias=attention_bias,
        )
        self.FFN_layer_1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer_2 = nn.Linear(out_dim * 2, out_dim)

        if batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(out_dim)
            self.batch_norm_2 = nn.BatchNorm1d(out_dim)

        if layer_norm:
            self.layer_norm_1 = nn.LayerNorm(out_dim)
            self.layer_norm_2 = nn.LayerNorm(out_dim)


    def forward(self, h, A, lengths, alpha):
        """
        Computes a GTA3 forward pass.

        Args:
          h:     node embeddings [batch, nodes, in_dim] or [nodes, in_dim]
          A:     matrix provided to the weighting function [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]
          alpha: value used by the weighting function

        """
        h_in = h

        # perform multihead attention
        h, log_dict = self.aa_attention(h, A, lengths, alpha)
        h = einops.rearrange(h, 'b k n d -> b n (k d)', d=self.out_dim//self.num_heads)

        # TODO: Check against ground truth
        # print("Embeddings after attention in batched model")
        # for i,h_i in enumerate(h):
        #     print(f"Embedding sum {i}: {h_i.sum()}")

        # apply the O matrix
        h = self.O(h)
        
        # residual & normalization
        if self.residual_heads:
            h = h_in + h
        if self.batch_norm:
            h = h.moveaxis(-1,-2)
            h = self.batch_norm_1(h) # TODO: check that this is correct
            h = h.moveaxis(-2,-1)
        if self.layer_norm:
            h = self.layer_norm_1(h)
        
        # feed forward network
        h_tmp = h
        h = self.FFN_layer_1(h)
        h = F.relu(h)
        h = F.dropout(h)
        h = self.FFN_layer_2(h)

        # residual & normalization
        if self.residual_ffn:
            h = h_tmp + h
        if self.batch_norm:
            h = h.moveaxis(-1,-2)
            h = self.batch_norm_2(h) # TODO: check that this is correct
            h = h.moveaxis(-2,-1)
        if self.layer_norm:
            h = self.layer_norm_2(h)
        return h, log_dict
    

class GTA3BaseModel(L.LightningModule):
    score_direction = None
    score_name = None
    
    def __init__(self, model_params, train_params):
        """
        Graph Transformer with Adjacency Aware Attention (GTA3) base model.

        Args:
          model_params: dictionary containing the model parameters
          train_params: dictionary containing the training parameters

        """
        super().__init__()

        self.train_params = train_params

        # initialize the alpha value
        n_layers = model_params['num_layers']
        if isinstance(model_params['alpha_init'], str):
            if model_params['alpha_init'] == 'linear_inc_dec':
                lb, ub = model_params['alpha_init_kwargs']['lb'], model_params['alpha_init_kwargs']['ub']
                if n_layers % 2 == 0:
                    a = torch.linspace(lb, ub, n_layers//2)
                    alpha = torch.cat([a, a.flip(0)])
                else:
                    a = torch.linspace(lb, ub, n_layers//2+1)
                    alpha = torch.cat([a[:-1], a.flip(0)])
            elif model_params['alpha_init'] == 'linear_inc':
                lb, ub = model_params['alpha_init_kwargs']['lb'], model_params['alpha_init_kwargs']['ub']
                alpha = torch.linspace(lb, ub, n_layers)
            elif model_params['alpha_init'] == 'linear_dec':
                lb, ub = model_params['alpha_init_kwargs']['lb'], model_params['alpha_init_kwargs']['ub']
                alpha = torch.linspace(ub, lb, n_layers)
        else:
            try:
                alpha = torch.tensor(model_params['alpha_init'], dtype=torch.float)
            except:
                raise ValueError("Invalid alpha_init parameter!", model_params['alpha_init'])

        if model_params['alpha'] == 'per_model' and alpha.numel() != 1:
            raise ValueError(f"Invalid alpha_init parameter (because of `per_model`)! Expected 1 value, got {alpha.numel()}!")
        elif model_params['alpha'] == 'per_layer':
            try:
                alpha = alpha.expand(n_layers).clone()  # clone necessary (torch is lazy o/w)
            except:
                raise ValueError(f"Invalid alpha_init parameter (because of `per_layer`)! cannot expand {alpha.shape} to ({n_layers},)!")
        elif model_params['alpha'] == 'per_head':
            raise NotImplementedError("per head alpha not implemented")
        
        if model_params['alpha'] == 'fixed':
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = torch.nn.Parameter(alpha)
            if model_params['alpha_init'] == 'per_layer' and self.alpha.numel() != n_layers:
                raise ValueError(f"Invalid alpha_init parameter (because of `per_layer`)! Expected {n_layers} values, got {self.alpha.numel()}!")
            elif model_params['alpha_init'] == 'per_head' and self.alpha.numel() != model_params['num_heads'] * n_layers:
                raise ValueError(f"Invalid alpha_init parameter (because of `per_head`)! Expected {model_params['num_heads'] * n_layers} values, got {self.alpha.numel()}!")
        print(f"alpha: {self.alpha}")
        
        self.per_layer_alpha = self.alpha.dim() >= 1

        # creates an embedding depending on the node type
        self.embedding = nn.Embedding(model_params['num_in_types'], model_params['hidden_dim'])
        
        # positional embeddings
        self.use_pos_enc = False
        if model_params['pos_encoding'] == 'laplacian':
            self.use_pos_enc = True
            self.pos_embedding = nn.Linear(model_params['pos_enc_dim'], model_params['hidden_dim'])
        elif model_params['pos_encoding'] == 'wl':
            self.use_pos_enc = True
            self.pos_embedding = nn.Embedding(model_params['max_num_nodes'], model_params['hidden_dim'])
        elif model_params['pos_encoding'] != 'none':
            raise ValueError(f"Unknown positional encoding parameter '{model_params['pos_encoding']}'!")

        # the main part of the model
        self.gta3_layers = nn.ModuleList(
            [ GTA3Layer(
                in_dim=model_params['hidden_dim'], out_dim=model_params['hidden_dim'], num_heads=model_params['num_heads'], phi=model_params['phi'],
                residual=model_params['residual'], batch_norm=model_params['batch_norm'], layer_norm=model_params['layer_norm'], 
                attention_bias=model_params['attention_bias']) 
            for _ in range(model_params['num_layers']-1) ]
        )
        self.gta3_layers.append(
            GTA3Layer(
                in_dim=model_params['hidden_dim'], out_dim=model_params['out_dim'], num_heads=model_params['num_heads'], phi=model_params['phi'],
                residual=model_params['residual'], batch_norm=model_params['batch_norm'], layer_norm=model_params['layer_norm'],
                attention_bias=model_params['attention_bias'])
        )


    def training_step(self, batch, batch_idx):
        raise NotImplementedError("GTA3BaseModel: Define training_step function!")


    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("GTA3BaseModel: Define validation_step function!")

    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError("GTA3BaseModel: Define test_step function!")


    def configure_optimizers(self):
        if not hasattr(self, "score_direction") or self.score_direction is None:
            raise NotImplementedError("GTA3BaseModel: Define score_direction attribute! (min|max)")
        if not hasattr(self, "score_name") or self.score_name is None:
            raise NotImplementedError("GTA3BaseModel: Define score_name attribute! (e.g. 'val_loss')")
        
        default_params = [p for p in self.parameters() if p is not self.alpha]
        param_groups = [
            {'params': default_params},
        ]
        if isinstance(self.alpha, torch.nn.Parameter):
            param_groups.append({'params': self.alpha, 'lr': self.train_params['alpha_lr']})

        optimizer = optim.Adam(
            param_groups,
            lr=self.train_params['lr'],
            weight_decay=self.train_params['weight_decay']
        )
        print(f"Optimizer: {optimizer}")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.score_direction,
            factor=self.train_params['lr_reduce_factor'],
            patience=self.train_params['lr_schedule_patience'],
            verbose=True
        )
        print(f"LR-Scheduler: {scheduler}")
        return (
            [optimizer],
            [{'scheduler': scheduler, 'interval': 'epoch', 'monitor': self.score_name, 'strict': False}]
        )