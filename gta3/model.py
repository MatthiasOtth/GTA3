from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


def phi_no_weighting(a, A):
    return a


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
        print(self.sqrt_out_dim)

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=bias)

        self.softmax = nn.Softmax(dim=-2)
        self.phi = phi


    def forward(self, h, A):
        """
        Computes an adjacency aware multi head attention forward pass.

        Args:
          h: node embeddings [batch, nodes, in_dim] or [nodes, in_dim]
          A: matrix provided to the weighting function [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]

        """
        assert len(h.shape) in (2,3), f"AdjacencyAwareMultiHeadAttention Error: Got invalid shape for h {h.shape}!"

        # apply the query, key and value matrices
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        if len(h.shape) == 2:
            Q_h = einops.rearrange(Q_h, 'n (k d) -> k n d', d=self.out_dim)
            K_h = einops.rearrange(K_h, 'n (k d) -> k n d', d=self.out_dim)
            V_h = einops.rearrange(V_h, 'n (k d) -> k n d', d=self.out_dim)
        else:
            Q_h = einops.rearrange(Q_h, 'b n (k d) -> b k n d', d=self.out_dim)
            K_h = einops.rearrange(K_h, 'b n (k d) -> b k n d', d=self.out_dim)
            V_h = einops.rearrange(V_h, 'b n (k d) -> b k n d', d=self.out_dim)
        
        # compute the dot products
        attention = torch.matmul(Q_h, K_h.transpose(-1,-2)).transpose(-1,-2)

        # apply scaling and softmax to attention
        attention = self.softmax(attention / self.sqrt_out_dim)

        # reweight using the adjacency information
        attention = self.phi(attention, A)

        # sum value tensors scaled by the attention weights
        h_heads = torch.matmul(attention.transpose(-1,-2), V_h)

        return h_heads


class GTA3Layer(nn.Module):

    def __init__(self, in_dim, out_dim, phi, num_heads=8, residual=True, batch_norm=True, attention_bias=True):
        """
        Graph Transformer with Adjacency Aware Attention (GTA3).

        Args:
          in_dim:         the dimension of the input vectors
          out_dim:        the dimension of the output vectors
          phi:            the weighting function to be used
          num_heads:      the number of attention heads to be used (default: 8)
          residual:       whether to use residual connections (default: True) (note that if in_dim != out_dim the heads will not have a residual connection)
          batch_norm:     whether to use batch normalization (default: True)
          attention_bias: whether the attention matrices (Q, K, V) use a bias or not (default: True)

        """
        super().__init__()

        self.out_dim = out_dim
        self.batch_norm = batch_norm
        self.residual_heads = residual and in_dim == out_dim
        self.residual_ffn = residual

        if phi == 'id':
            self.phi = phi_no_weighting
        else:
            print(f"GTA3 Error: Unknown phi function {phi}! Use one of the following: 'id'")
            exit()

        self.O = nn.Linear(out_dim * num_heads, out_dim)
        self.aa_attention = AdjacencyAwareMultiHeadAttention(in_dim=in_dim, out_dim=out_dim, phi=self.phi, num_heads=num_heads, bias=attention_bias)
        self.FFN_layer_1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer_2 = nn.Linear(out_dim * 2, out_dim)

        if batch_norm:
            self.batch_norm_1 = nn.BatchNorm1d(out_dim)
            self.batch_norm_2 = nn.BatchNorm1d(out_dim)


    def forward(self, h, A):
        """
        Computes a GTA3 forward pass.

        Args:
          h: node embeddings [batch, nodes, in_dim] or [nodes, in_dim]
          A: matrix provided to the weighting function [batch, num_nodes, num_nodes] or [num_nodes, num_nodes]

        """
        h_in = h

        # perform multihead attention
        h = self.aa_attention(h, A)
        if len(h.shape) == 3: h = einops.rearrange(h, 'k n d -> n (k d)', d=self.out_dim)
        else:                 h = einops.rearrange(h, 'b k n d -> b n (k d)', d=self.out_dim)

        # apply the O matrix
        h = self.O(h)

        # residual & normalization
        if self.residual_heads:
            h = h_in + h
        if self.batch_norm:
            h = self.batch_norm_1(h) # TODO: check that this is correct

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
            h = self.batch_norm_2(h) # TODO: check that this is correct

        return h