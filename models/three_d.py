# -*- coding: utf-8 -*-
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from config import MODAL_DIM, THREE_D_TRANSFORMER_LAYERS, THREE_D_EMBED_DIM

class GaussianSmearing(nn.Module):
    def __init__(self, start=0.0, stop=10.0, num_gaussians=64):
        super().__init__()
        offset = tc.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist_mat):
        dist_mat = dist_mat.unsqueeze(-1)
        return tc.exp(self.coeff * (dist_mat - self.offset)**2)


class ThreeDTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, rbf_dim=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.rbf_encoder = GaussianSmearing(start=0.0, stop=10.0, num_gaussians=rbf_dim)
        self.rbf_proj = nn.Sequential(
            nn.Linear(rbf_dim, num_heads),
            nn.GELU(),
            nn.Linear(num_heads, num_heads)
        )
        self.w_o = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(), 
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, dist_mat, mask=None):
        B, L, _ = x.shape
        residual = x
        Q = self.w_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2) 
        K = self.w_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = tc.matmul(Q, K.transpose(-2, -1)) * self.scale 
        rbf_feat = self.rbf_encoder(dist_mat)
        rbf_bias = self.rbf_proj(rbf_feat).permute(0, 3, 1, 2)
        attn_scores = attn_scores + rbf_bias
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = tc.matmul(attn_weights, V) 
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.w_o(output)
        x = self.norm1(residual + output)
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm2(residual + x)
        return x


class ThreeDTransformer(nn.Module):
    def __init__(self, atom_dim=10, embed_dim=128, num_layers=4, heads=8, dropout=0.5):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, embed_dim)
        self.layers = nn.ModuleList([
            ThreeDTransformerLayer(embed_dim, heads, dropout) for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(tc.zeros(1, 1, embed_dim))
        self.proj_out = nn.Linear(embed_dim, MODAL_DIM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_feats, coords):
        if atom_feats.dim() == 2:
            atom_feats = atom_feats.unsqueeze(0) 
            coords = coords.unsqueeze(0) 
        B, L, _ = atom_feats.shape
        x = self.atom_embed(atom_feats) 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = tc.cat((cls_tokens, x), dim=1) 
        dist_mat = tc.cdist(coords, coords, p=2) 
        dist_mat_padded = F.pad(dist_mat, (1, 0, 1, 0), "constant", 0) 
        for layer in self.layers:
            x = layer(x, dist_mat_padded)
        cls_out = x[:, 0, :] 
        return self.proj_out(cls_out)