# -*- coding: utf-8 -*-
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModel
from config import (
    BERT_PATH, MODAL_DIM, HEADS, DROPOUT, 
    GAT_LAYERS, HIDDEN_DIM, BERT_UNFREEZE_LAYERS,
    THREE_D_TRANSFORMER_LAYERS, THREE_D_EMBED_DIM
)
from .attention import CrossModalAttention, GatedFusion
from .three_d import ThreeDTransformer
from .gat import HierarchicalGATModel


class MolModel(nn.Module):
    def __init__(self, fp_dim, task_type="binary", num_labels=1, 
                 hidden_dim=128, gat_layers=4, modal_dim=256, 
                 heads=8, dropout=0.5):
        super().__init__()
        self.task_type = task_type
        
        print(f"[INFO] ChemBERTa : {BERT_PATH}")
        try:
            self.bert = AutoModel.from_pretrained(
                BERT_PATH,
                local_files_only=True,
                use_safetensors=False,
                trust_remote_code=False,
            )
            print(f"[SUCCESS] ChemBERTa | hidden_size = {self.bert.config.hidden_size}")
        except Exception as e:
            print(f"[FATAL] ChemBERTa load failed: {e}")
            raise e

        
        for i, param in enumerate(self.bert.encoder.layer):
            if i < len(self.bert.encoder.layer) - BERT_UNFREEZE_LAYERS:
                for p in param.parameters():
                    p.requires_grad = False

        bert_hidden_size = self.bert.config.hidden_size
        
        self.lstm = nn.LSTM(bert_hidden_size, modal_dim // 2, 
                           num_layers=1, bidirectional=True, 
                           batch_first=True, dropout=dropout)
        
        self.bert_proj = nn.Sequential(
            nn.Linear(modal_dim, modal_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.LayerNorm(modal_dim)
        )
        
        self.fp_att = nn.Sequential(
            nn.Linear(fp_dim, modal_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.LayerNorm(modal_dim)
        )
        
       
        self.graph_proj = nn.Sequential(
            nn.Linear(1, modal_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.LayerNorm(modal_dim)
        )
        
        self.gin_model = HierarchicalGATModel(
            in_dim=21, 
            hidden_dim=hidden_dim, 
            out_dim=modal_dim,
            gat_layers=gat_layers, 
            heads=heads, 
            dropout=dropout
        )
        
        self.three_d_model = ThreeDTransformer(
            atom_dim=10, 
            embed_dim=THREE_D_EMBED_DIM, 
            num_layers=THREE_D_TRANSFORMER_LAYERS,
            heads=heads, 
            dropout=dropout
        )
        
        self.three_d_proj = nn.Sequential(
            nn.Linear(MODAL_DIM, modal_dim), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.LayerNorm(modal_dim)
        )
        
        self.cross_attn_graph_to_seq = CrossModalAttention(modal_dim, heads=heads, dropout=dropout)
        self.cross_attn_seq_to_graph = CrossModalAttention(modal_dim, heads=heads, dropout=dropout)
        
        D = modal_dim
        self.proj_td = nn.Sequential(nn.Linear(MODAL_DIM, D), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(D))
        self.proj_graph = nn.Sequential(nn.Linear(modal_dim, D), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(D))
        self.proj_fp = nn.Sequential(nn.Linear(modal_dim, D), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(D))
        
        self.structural_fusion = GatedFusion(input_dim=D)
        
        final_fusion_input_dim = modal_dim + D
        self.fusion_mlp = nn.Sequential(
            nn.Linear(final_fusion_input_dim, 512), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.LayerNorm(512)
        )
        self.out_proj = nn.Sequential(
            nn.Linear(512, 256), 
            nn.GELU(), 
            nn.Dropout(dropout), 
            nn.LayerNorm(256)
        )
        self.out = nn.Linear(256, num_labels)
        self.uncertainty_head = nn.Linear(256, num_labels)
        self.bert_classifier = nn.Linear(modal_dim, num_labels) 
        self.graph_classifier = nn.Linear(modal_dim, num_labels) 

    def forward(self, input_ids, attention_mask, fp, graph_feats, 
                coords_list, adj_list, atom_feats_list, chain_masks_list):
        
        bert_out_full = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        lstm_out, _ = self.lstm(bert_out_full)
        bert_pooled = lstm_out.mean(dim=1) 
        bert_feat = self.bert_proj(bert_pooled) 
        
        fp_feat = self.fp_att(fp) 
        graph_desc = self.graph_proj(graph_feats)   # dummy 
        
        # GAT + Graph Processing
        gin_vecs = []
        B = input_ids.size(0)
        device = input_ids.device
        for idx in range(B):
            adj = adj_list[idx]
            atom_feats = atom_feats_list[idx]
            chain_mask = chain_masks_list[idx]
            coords = coords_list[idx]
            if atom_feats.shape[0] == 0:
                gin_vecs.append(tc.zeros(MODAL_DIM, dtype=tc.float32, device=device))
                continue
            if isinstance(adj, np.ndarray):
                adj = tc.from_numpy(adj).to(device)
            adj = adj.float()
            edge_index = adj.nonzero().t().contiguous()
            gin_vec = self.gin_model(atom_feats, edge_index, chain_mask, coords)
            gin_vecs.append(gin_vec)
        gin_batch = tc.stack(gin_vecs, dim=0) 
        
        # 3D Transformer
        three_d_vecs = []
        for idx in range(B):
            atom_feats = atom_feats_list[idx]
            coords = coords_list[idx]
            if atom_feats.shape[0] == 0:
                three_d_vecs.append(tc.zeros(MODAL_DIM, dtype=tc.float32, device=device))
                continue
            td_vec = self.three_d_model(atom_feats, coords)
            if td_vec.dim() == 2:
                td_vec = td_vec.squeeze(0)
            three_d_vecs.append(td_vec)
        three_d_batch = tc.stack(three_d_vecs, dim=0) 
        td_feat = self.three_d_proj(three_d_batch) 
        
        # Cross Modal Attention
        bert_seq = bert_feat.unsqueeze(1) 
        graph_seq = gin_batch.unsqueeze(1) 
        updated_bert = self.cross_attn_graph_to_seq(query=bert_seq, key_value=graph_seq).squeeze(1)
        updated_graph = self.cross_attn_seq_to_graph(query=graph_seq, key_value=bert_seq).squeeze(1)
        
        bert_updated = bert_feat + updated_bert
        graph_updated = gin_batch + updated_graph
        
        # Gated Fusion
        td_proj = self.proj_td(td_feat) 
        graph_proj = self.proj_graph(graph_updated) 
        fp_proj = self.proj_fp(fp_feat) 
        structural_embedding = self.structural_fusion(graph_proj, td_proj, fp_proj)
        
        final_embedding = tc.cat([bert_updated, structural_embedding], dim=-1)
        h = self.fusion_mlp(final_embedding)
        h = self.out_proj(h)
        mean_out = self.out(h)
        
        bert_logits = self.bert_classifier(bert_updated) 
        graph_logits = self.graph_classifier(graph_updated) 
        
        if self.task_type == "regression":
            log_var = self.uncertainty_head(h)
            return mean_out, log_var, td_proj, graph_proj, fp_proj, bert_updated, bert_logits, graph_logits
        
        if mean_out.dim() == 1:
            mean_out = mean_out.unsqueeze(-1)
        return mean_out, td_proj, graph_proj, fp_proj, bert_updated, bert_logits, graph_logits