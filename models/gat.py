# -*- coding: utf-8 -*-
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import HIDDEN_DIM
from utils import torch_safe_nan_to_num

class HierarchicalGATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, gat_layers=3, heads=4, dropout=0.3):
        super().__init__()
        self.atom_dim = 10
        self.coord_proj = nn.Linear(3, self.atom_dim)
        self.proj_in = nn.Linear(21, hidden_dim)   # atom(10) + coord(3) + chain(1) = 14 
        self.gat_layers = nn.ModuleList()
        self.dist_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // heads),
            nn.ReLU(),
            nn.Linear(hidden_dim // heads, hidden_dim // heads)
        )
        for i in range(gat_layers):
            in_h = hidden_dim if i == 0 else hidden_dim
            out_h = hidden_dim // heads
            self.gat_layers.append(GATConv(in_h, out_h, heads=heads, dropout=dropout, edge_dim=hidden_dim // heads))
            self.gat_layers.append(nn.Linear(out_h * heads, hidden_dim))
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.hier_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, chain_mask, coords):
        if x.numel() == 0 or coords.numel() == 0 or edge_index.numel() == 0:
            return tc.zeros(self.proj_out.out_features, dtype=x.dtype, device=x.device)
        
        coord_feats = self.coord_proj(coords)
        chain_mask_exp = chain_mask.unsqueeze(-1).to(x.device)
        
        if x.shape[0] != coord_feats.shape[0] or x.shape[0] != chain_mask_exp.shape[0]:
            return tc.zeros(self.proj_out.out_features, dtype=x.dtype, device=x.device)
        
        x = tc.cat([x, coord_feats, chain_mask_exp], dim=-1)
        x = self.proj_in(x)
        
        dists = tc.norm(coords[edge_index[0]] - coords[edge_index[1]], dim=-1, keepdim=True)
        dists = torch_safe_nan_to_num(dists, device=x.device, dtype=x.dtype)
        edge_attr = self.dist_encoder(dists)
        
        for i in range(0, len(self.gat_layers), 2):
            conv_layer = self.gat_layers[i]
            proj_layer = self.gat_layers[i+1]
            x_residual = x
            x = conv_layer(x, edge_index, edge_attr=edge_attr)
            x = proj_layer(x)
            x = F.elu(x) + x_residual
            x = self.hier_norm(x)
        
        return self.proj_out(x.mean(dim=0))