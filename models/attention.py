import torch as tc
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.5):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

    def forward(self, query, key_value):
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return attn_output

class GatedFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        combined_dim = input_dim * 3
        self.gate_gat = nn.Linear(combined_dim, input_dim)
        self.gate_3d = nn.Linear(combined_dim, input_dim)
        self.gate_fp = nn.Linear(combined_dim, input_dim)
        self.mlp_gat = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU())
        self.mlp_3d = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU())
        self.mlp_fp = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU())

    def forward(self, gat_feat, td_feat, fp_feat):
        gat_feat_aligned = self.mlp_gat(gat_feat)
        td_feat_aligned = self.mlp_3d(td_feat)
        fp_feat_aligned = self.mlp_fp(fp_feat)
        combined = tc.cat([gat_feat_aligned, td_feat_aligned, fp_feat_aligned], dim=-1)
        weight_gat = tc.sigmoid(self.gate_gat(combined))
        weight_3d = tc.sigmoid(self.gate_3d(combined))
        weight_fp = tc.sigmoid(self.gate_fp(combined))
        fused_feat = (weight_gat * gat_feat_aligned + weight_3d * td_feat_aligned + weight_fp * fp_feat_aligned)
        return fused_feat