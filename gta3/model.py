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

def phi_alpha_pow_dist(a, A, alpha):
    alpha = torch.clamp(alpha, min=0) #, max=1)
    new_a = torch.pow(alpha + 1e-10, A) * a
    new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    new_a = F.normalize(new_a, p=1, dim=-1)
    return new_a

def phi_inverse_hops(a, A, alpha):
    # Assumes that A is shortest path matrix
    if alpha == 0:
        new_a = torch.where(A==1, a, torch.zeros_like(a))
    else:
        x = torch.clamp(1/torch.abs(alpha), max=10)
        #TODO: Rewrite so we don't need epsilon
        new_a = 1./(torch.pow(A, x) * a + 1e-5)
        new_a = torch.where(A==0, torch.zeros_like(a), new_a)
    new_a = F.normalize(new_a, p=1, dim=-1)
    # new_a = F.softmax(new_a, dim=-1)

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
        attention = torch.matmul(Q_h, K_h.transpose(-1,-2))
          
        #TODO: Optimize generation of mask
        mask = torch.full((attention.shape[0], attention.shape[2]), False, device=attention.device)
        for i in range(lengths.shape[0]):
            mask[i,lengths[i]:] = True
        mask.unsqueeze_(1)
        mask.unsqueeze_(2)
        
        attention = attention.masked_fill(mask, float('-inf'))
        # apply scaling and softmax to attention
        attention = self.softmax(attention / self.sqrt_out_dim)

        mask = mask.moveaxis(-2,-1)
        attention = attention.masked_fill(mask, float(0))
       
        # reweight using the adjacency information
        attention = attention.moveaxis(1,0)
        log_dict = {}
        log_dict["attention/attention_pre_transform_d1"]  = attention[:,A==1].mean(), attention[:,A==1].reshape(-1).shape[0]
        log_dict["attention/attention_pre_transform_d2+"] = attention[:,A >1].mean(), attention[:,A >1].reshape(-1).shape[0]

        attention = self.phi(attention, A, alpha)

        log_dict["attention/attention_post_transform_d1"]  = attention[:,A==1].mean(), attention[:,A==1].reshape(-1).shape[0]
        log_dict["attention/attention_post_transform_d2+"] = attention[:,A >1].mean(), attention[:,A >1].reshape(-1).shape[0]
        attention = attention.moveaxis(0,1)
        
        # sum value tensors scaled by the attention weights
        h_heads = torch.matmul(attention, V_h)
        return h_heads, log_dict


class GTA3Layer(nn.Module):

    def __init__(self, in_dim, out_dim, phi, num_heads=8, residual=True, batch_norm=False, layer_norm=True, attention_bias=True):
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
        elif phi == 'test':
            self.phi = phi_simple_adj_weighting
        elif phi == 'inverse_hops':
            self.phi = phi_inverse_hops
        elif phi == 'alpha_pow_dist':
            self.phi = phi_alpha_pow_dist
        else:
            print(f"GTA3 Error: Unknown phi function {phi}! Use one of the following: 'none', 'test'")
            exit()

        self.O = nn.Linear(out_dim, out_dim)
        self.aa_attention = AdjacencyAwareMultiHeadAttention(in_dim=in_dim, out_dim=out_dim//num_heads, phi=self.phi, num_heads=num_heads, bias=attention_bias)
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
        if model_params['alpha'] == 'fixed':
            self.per_layer_alpha = False
            self.alpha = torch.tensor([model_params['alpha_init']], dtype=torch.float)
        elif model_params['alpha'] == 'per_model': # TODO: test this...
            self.per_layer_alpha = False
            self.alpha = torch.nn.Parameter(torch.tensor([model_params['alpha_init']], dtype=torch.float))
        elif model_params['alpha'] == 'per_layer': # TODO: test this...
            self.per_layer_alpha = True
            if type(model_params['alpha_init']) == list:
                assert len(model_params['alpha_init']) == model_params['num_layers'], \
                    f"Number of layers ({model_params['num_layers']}) does not match number of initial alpha values given ({len(model_params['alpha_init'])})!"
                self.alpha = torch.nn.Parameter(torch.tensor(model_params['alpha_init'], dtype=torch.float))
            else:
                self.alpha = torch.nn.Parameter(torch.tensor([model_params['alpha_init'] for _ in range(model_params['num_layers'])], dtype=torch.float))
        elif model_params['alpha'] == 'per_head': # TODO: test this...
            self.per_layer_alpha = True
            # TODO: implement
            raise NotImplementedError("alpha per head is not yet implemented...")
        else:
            raise ValueError("Invalid alpha model parameter!", model_params['alpha'])

        # creates an embedding depending on the node type
        self.embedding = nn.Embedding(model_params['num_in_types'], model_params['hidden_dim'])
        
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
        raise NotImplementedError("GTA3BaseModel: Define training_step function!")


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.train_params['lr'])
        return optimizer