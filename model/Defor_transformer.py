# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class DeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_points=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, reference_points, value):
        B, N, _ = query.shape
        _, Nv, _ = value.shape
        H = W = int(Nv ** 0.5)
        
        offsets = self.sampling_offsets(query).view(
            B, N, self.num_heads, self.num_points, 2)
        
        attn_weights = self.attention_weights(query).view(
            B, N, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        sampling_locations = reference_points.unsqueeze(2).unsqueeze(2) + offsets
        sampling_locations = sampling_locations * 2 - 1

        value = self.value_proj(value)
        value = value.view(B, Nv, self.num_heads, self.head_dim)
        value = value.permute(0, 2, 3, 1)

        grid = sampling_locations.permute(0, 2, 1, 3, 4).reshape(B*self.num_heads, N, self.num_points, 2)
        
        value = value.reshape(B * self.num_heads, self.head_dim, H, W)

        sampled_values = F.grid_sample(
            value,
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )

        sampled_values = sampled_values.reshape(
            B, self.num_heads, self.head_dim, N, self.num_points)
        sampled_values = sampled_values.permute(0, 3, 1, 4, 2) 

        weighted_values = torch.einsum('bnhk,bnhkd->bnhd', attn_weights, sampled_values)
        
        output = weighted_values.reshape(B, N, self.embed_dim)
        output = self.output_proj(output)
        return output

class DeformableTransformerLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_points=4, dropout = 0.1):
        super().__init__()

        self.self_attn = DeformableAttention(d_model, n_heads, n_points)
        self.ref_point_proj = nn.Linear(d_model, 2)

        self.activation = nn.ReLU()
        
    def forward_ffn(self, src):
        src2 = self.dropout2(self.activation(self.linear1(src)))
        src = src + src2
        return src

    def forward(self, query, value):
        reference_points = self.ref_point_proj(query).sigmoid()
        attn_output = self.self_attn(query, reference_points, value)
        return attn_output



class DeformableTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.encoder1 = DeformableTransformerLayer(d_model = dim)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, query, value, pos=None):
        x = query + pos
        
        memory = self.encoder1(x, value)
        
        return memory

    
    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(256)

    def forward(self, tgt, memory, query_pos, reference_points):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, query_pos, reference_points)

        output = self.norm(output)


        return output



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
