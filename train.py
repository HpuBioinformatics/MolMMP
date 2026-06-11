# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.deprecation')

import os, time
import numpy as np
import torch as tc
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve
from config import *
from utils import set_seed, torch_safe_nan_to_num, seed_worker   # ← 这里确保有 seed_worker
from losses import FocalLoss, FocalLossWithPosWeight, ContrastiveLoss, UncertaintyWeightedLoss
from models.main_model import MolModel
from dataset import SmilesDataset, collate_fn
from features.graph import compute_fp_and_descriptors
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_loss_fn_for_training(task_type, y_train=None, gamma=GAMMA, alpha=ALPHA, dataset_name=None):
    if task_type == "binary":
        config = DATASET_SPECIFIC_CONFIG.get(dataset_name, DEFAULT_CONFIG)
        alpha = config["focal_alpha"]
        gamma = config["focal_gamma"]
        pos_ratio = None
        if y_train is not None:
            valid_y_train = y_train[~np.isnan(y_train)]
            if len(valid_y_train) > 0:
                pos_ratio = float(np.mean(valid_y_train))
        use_pos_weight_flag = bool(config.get("use_pos_weight", False))
        if use_pos_weight_flag and pos_ratio is not None and 0 < pos_ratio < 1:
            pos_weight = (1 - pos_ratio) / pos_ratio
            return FocalLossWithPosWeight(gamma=gamma, alpha=alpha, pos_weight=pos_weight)
        else:
            return FocalLoss(gamma=gamma, alpha=alpha)
    if task_type == "multilabel":
        return FocalLoss(gamma=gamma, alpha=alpha)
    if task_type == "regression":
        return UncertaintyWeightedLoss()
    return nn.CrossEntropyLoss()


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, task_type="binary",
                   scaler=None, contrastive_loss_fn=None, pretrain_mode=False, epoch=0):
    model.train()
    running_loss = 0.0
    n = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        fp = batch["fp"].to(DEVICE)
        graph_feats = batch["graph_feats"].to(DEVICE)
       
        original_labels = batch["labels"].to(DEVICE)
        mask = ~tc.isnan(original_labels)
        labels = tc.nan_to_num(original_labels, nan=0.0)
       
        if task_type in ["binary", "multilabel"]:
            labels = labels.clamp(0.0, 1.0)
           
        coords_list = [x.to(DEVICE) for x in batch["coords_list"]]
        adj_list = [x.to(DEVICE) for x in batch["adj_list"]]
        atom_feats_list = [x.to(DEVICE) for x in batch["atom_feats_list"]]
        chain_masks_list = [x.to(DEVICE) for x in batch["chain_masks_list"]]
       
        with tc.amp.autocast('cuda', enabled=(scaler is not None)):
            if task_type == "regression":
                out, log_var, td_out, graph_out, fp_out, bert_out, bert_logits, graph_logits = model(
                    input_ids, attention_mask, fp, graph_feats,
                    coords_list, adj_list, atom_feats_list, chain_masks_list
                )
            else:
                out, td_out, graph_out, fp_out, bert_out, bert_logits, graph_logits = model(
                    input_ids, attention_mask, fp, graph_feats,
                    coords_list, adj_list, atom_feats_list, chain_masks_list
                )
           
            out = tc.where(tc.isfinite(out), out, tc.zeros_like(out))
           
            if pretrain_mode:
                main_loss = tc.tensor(0.0, device=DEVICE)
                gat_fp_loss = contrastive_loss_fn(graph_out, fp_out)
                bert_gat_loss = contrastive_loss_fn(bert_out, graph_out)
                contrastive_loss = gat_fp_loss + bert_gat_loss
                total_loss = contrastive_loss
            else:
                if task_type in ["binary", "multilabel"]:
                    if labels.dim() == 1:
                        labels = labels.view(-1, 1)
                        mask = mask.view(-1, 1)
                    main_loss = loss_fn(out, labels, mask=mask)
                elif task_type == "regression":
                    main_loss = loss_fn(out.squeeze(-1), log_var.squeeze(-1), labels.squeeze(-1), mask=mask.squeeze(-1))
                else:
                    main_loss = loss_fn(out, labels.long().squeeze(-1))
               
                total_loss = main_loss
                if contrastive_loss_fn is not None and USE_CONTRASTIVE_REG:
                    gat_fp_reg_loss = contrastive_loss_fn(graph_out, fp_out)
                    bert_gat_reg_loss = contrastive_loss_fn(bert_out, graph_out)
                    contrast_reg_loss = gat_fp_reg_loss + bert_gat_reg_loss
                    total_loss += CONTRAST_LAMBDA * contrast_reg_loss
                   
                if not pretrain_mode and task_type in ["binary", "multilabel"]:
                    if bert_logits.dim() == 1:
                        bert_logits = bert_logits.view(-1, 1)
                    if graph_logits.dim() == 1:
                        graph_logits = graph_logits.view(-1, 1)
                  
                    bert_loss = loss_fn(bert_logits, labels, mask=mask)
                    graph_loss = loss_fn(graph_logits, labels, mask=mask)
                  
                    AUX_WEIGHT = 0.2
                    total_loss = total_loss + AUX_WEIGHT * (bert_loss + graph_loss)
        if not tc.isfinite(total_loss):
            print(f"[WARN] Non-finite loss at batch {batch_idx}. Skipping.")
            continue
           
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            tc.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            tc.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
           
        if scheduler is not None and pretrain_mode:
            scheduler.step()
           
        running_loss += float(total_loss.detach().cpu().item()) * input_ids.size(0)
        n += input_ids.size(0)
       
    return running_loss / max(n, 1)


def evaluate(model, loader, task_type="binary", dataset_name=None):
    model.eval()
    ys = []
    ps = []
    with tc.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            fp = batch["fp"].to(DEVICE)
            graph_feats = batch["graph_feats"].to(DEVICE)
            coords_list = [x.to(DEVICE) for x in batch["coords_list"]]
            adj_list = [x.to(DEVICE) for x in batch["adj_list"]]
            atom_feats_list = [x.to(DEVICE) for x in batch["atom_feats_list"]]
            chain_masks_list = [x.to(DEVICE) for x in batch["chain_masks_list"]]
            labels = batch["labels"]
          
            out, *_ = model(input_ids, attention_mask, fp, graph_feats,
                            coords_list, adj_list, atom_feats_list, chain_masks_list)
            out = tc.where(tc.isfinite(out), out, tc.zeros_like(out))
            ys.append(labels.cpu())
            ps.append(out.cpu())
          
    y_true_2d = tc.cat(ys, dim=0).numpy()
    y_pred_2d = tc.cat(ps, dim=0).numpy()
    prob_2d = 1 / (1 + np.exp(-y_pred_2d))
  
    metrics = {}
    if task_type in ["binary", "multilabel"]:
        if task_type == "multilabel" and dataset_name in ["tox21.csv", "clintox.csv"]:
            # micro + macro logic (kept from original)
            acc_list, auc_list, rec_list, f1_list, prec_list, ap_list = [], [], [], [], [], []
            all_y_true_list = []
            all_y_pred_bin_list = []
            all_y_prob_list = []
           
            num_tasks = y_true_2d.shape[1] if y_true_2d.ndim > 1 else 1
          
            for i in range(num_tasks):
                yt = y_true_2d[:, i] if y_true_2d.ndim > 1 else y_true_2d
                yp = prob_2d[:, i] if prob_2d.ndim > 1 else prob_2d
              
                valid_idx = ~np.isnan(yt)
                yt_valid = yt[valid_idx]
                yp_valid = yp[valid_idx]
              
                if len(yt_valid) == 0:
                    continue
              
                if len(np.unique(yt_valid)) > 1:
                    auc = roc_auc_score(yt_valid, yp_valid)
                    ap = average_precision_score(yt_valid, yp_valid)
                    fpr, tpr, thresholds = roc_curve(yt_valid, yp_valid)
                    j_scores = tpr - fpr
                    best_thresh = thresholds[np.argmax(j_scores)]
                else:
                    auc = float('nan')
                    ap = float('nan')
                    best_thresh = 0.5
              
                y_pr_valid = (yp_valid >= best_thresh).astype(int)
              
                auc_list.append(auc)
                ap_list.append(ap)
                acc_list.append(accuracy_score(yt_valid, y_pr_valid))
                rec_list.append(recall_score(yt_valid, y_pr_valid, zero_division=0))
                f1_list.append(f1_score(yt_valid, y_pr_valid, zero_division=0))
                prec_list.append(precision_score(yt_valid, y_pr_valid, zero_division=0))
              
                all_y_true_list.append(yt_valid)
                all_y_pred_bin_list.append(y_pr_valid)
                all_y_prob_list.append(yp_valid)
          
            if all_y_true_list:
                y_true_micro = np.concatenate(all_y_true_list)
                y_pred_micro = np.concatenate(all_y_pred_bin_list)
                y_prob_micro = np.concatenate(all_y_prob_list)
               
                if len(np.unique(y_true_micro)) > 1:
                    micro_auc = roc_auc_score(y_true_micro, y_prob_micro)
                    micro_ap = average_precision_score(y_true_micro, y_prob_micro)
                else:
                    micro_auc = micro_ap = float('nan')
               
                micro_acc = accuracy_score(y_true_micro, y_pred_micro)
                micro_rec = recall_score(y_true_micro, y_pred_micro, zero_division=0)
                micro_f1 = f1_score(y_true_micro, y_pred_micro, zero_division=0)
                micro_prec = precision_score(y_true_micro, y_pred_micro, zero_division=0)
            else:
                micro_auc = micro_ap = micro_acc = micro_rec = micro_f1 = micro_prec = float('nan')
           
            metrics.update({
                "ACC": float(np.nanmean(acc_list)) if acc_list else float('nan'),
                "AUC": float(np.nanmean(auc_list)) if auc_list else float('nan'),
                "Recall": float(np.nanmean(rec_list)) if rec_list else float('nan'),
                "F1": float(np.nanmean(f1_list)) if f1_list else float('nan'),
                "Precision": float(np.nanmean(prec_list)) if prec_list else float('nan'),
                "AP": float(np.nanmean(ap_list)) if ap_list else float('nan'),
                "ACC_micro": float(micro_acc),
                "AUC_micro": float(micro_auc),
                "Recall_micro": float(micro_rec),
                "F1_micro": float(micro_f1),
                "Precision_micro": float(micro_prec),
                "AP_micro": float(micro_ap),
            })
        else:
            valid_idx = ~np.isnan(y_true_2d.ravel())
            y_true_flat = y_true_2d.ravel()[valid_idx]
            prob_flat = prob_2d.ravel()[valid_idx]
          
            if len(np.unique(y_true_flat)) > 1:
                auc = roc_auc_score(y_true_flat, prob_flat)
                ap = average_precision_score(y_true_flat, prob_flat)
                fpr, tpr, thresholds = roc_curve(y_true_flat, prob_flat)
                best_thresh = thresholds[np.argmax(tpr - fpr)]
            else:
                auc = float('nan')
                ap = float('nan')
                best_thresh = 0.5
              
            pred_flat = (prob_flat >= best_thresh).astype(int)
          
            metrics.update({
                "ACC": float(accuracy_score(y_true_flat, pred_flat)),
                "AUC": float(auc),
                "Recall": float(recall_score(y_true_flat, pred_flat, zero_division=0)),
                "F1": float(f1_score(y_true_flat, pred_flat, zero_division=0)),
                "Precision": float(precision_score(y_true_flat, pred_flat, zero_division=0)),
                "AP": float(ap)
            })
    else:
        valid_idx = ~np.isnan(y_true_2d.ravel())
        y_pred_flat = y_pred_2d.ravel()[valid_idx]
        y_true_flat = y_true_2d.ravel()[valid_idx]
        mse = np.mean((y_true_flat - y_pred_flat) ** 2)
        metrics.update({
            "MSE": float(mse),
            "RMSE": float(np.sqrt(mse)),
            "MAE": float(np.mean(np.abs(y_true_flat - y_pred_flat)))
        })
    return metrics


def run():
    set_seed(RUN_SEED)
    import glob
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
   
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    results = {}
   
    for dataset, task_type in DATASETS.items():
        print(f"\n=== Dataset: {dataset} | Task: {task_type} ===")
       
        if dataset == "CTMTL/CTMTL.csv":
            full_path = "/home/zhaowanli/zhao/1-SynthMol/Data/CTMTL/CTMTL.csv"
        elif dataset in ["tox21.csv", "clintox.csv", "sider.csv"]:
            full_path = os.path.join(MOLECULENET_ROOT, dataset)
        else:
            full_path = os.path.join(DATA_ROOT, dataset)
           
        if not os.path.exists(full_path):
            print(f"[WARN] dataset file not found: {full_path}, skipping.")
            continue
        df = None
        for enc in ["utf-8", "latin1", "gbk"]:
            try:
                df = pd.read_csv(full_path, encoding=enc)
                print(f"[INFO] Successfully read {dataset} with encoding={enc}")
                break
            except Exception as e:
                print(f"[WARN] Failed to read with encoding={enc}: {e}")
        if df is None:
            raise RuntimeError(f"Failed to read CSV file: {full_path}")
        if dataset == "CTMTL/CTMTL.csv":
            print(f"[SPECIAL] Processing CTMTL dataset...")
            if "Canonical SMILES" not in df.columns or "Toxicity Value" not in df.columns:
                print(f"[ERROR] CTMTL missing required columns")
                continue
            smi_col = "Canonical SMILES"
            label_col = "Toxicity Value"
            if smi_col:
                def get_canonical(s):
                    m = Chem.MolFromSmiles(str(s))
                    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) if m else None
               
                df[smi_col] = df[smi_col].apply(get_canonical)
                initial_count = len(df)
                df = df.dropna(subset=[smi_col])
                df = df.drop_duplicates(subset=[smi_col])
                print(f"[CLEAN] {dataset}: {initial_count} -> {len(df)} samples after deduplication.")
            smiles = df[smi_col].astype(str).tolist()
            raw_labels = df[label_col].values
            try:
                labels = pd.to_numeric(raw_labels, errors='coerce').astype(np.float32)
            except:
                print(f"[ERROR] Cannot convert Toxicity Value to numeric")
                continue
            valid_mask = [isinstance(s, str) and s.strip() != '' and s.strip().lower() != 'nan' for s in smiles]
            smiles = [s.strip() for s, v in zip(smiles, valid_mask) if v]
            labels = labels[valid_mask]
            threshold = DATASET_SPECIFIC_CONFIG[dataset]["binary_threshold"]
            labels = (labels >= threshold).astype(np.float32)
            labels = labels.reshape(-1, 1)
            labels = np.nan_to_num(labels, nan=0.0)
            print(f"[INFO] CTMTL loaded: {len(smiles)} samples, positive: {(labels >= 0.5).sum()}")
        else:
            smi_col = next((c for c in df.columns if "smi" in c.lower() or "mol" in c.lower()), None)
            if smi_col:
                def get_canonical(s):
                    m = Chem.MolFromSmiles(str(s))
                    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) if m else None
               
                df[smi_col] = df[smi_col].apply(get_canonical)
                initial_count = len(df)
                df = df.dropna(subset=[smi_col])
                df = df.drop_duplicates(subset=[smi_col])
                print(f"[CLEAN] {dataset}: {initial_count} -> {len(df)} samples after deduplication.")
            if smi_col is None:
                print(f"[ERROR] no smiles column found in {full_path}")
                continue
            smiles = df[smi_col].tolist()
            labels_df = df.drop(columns=[smi_col])
            ignore_cols = ['mol_id', 'smiles', 'SMILES', 'ID', 'id']
            cols_to_drop = [c for c in ignore_cols if c in labels_df.columns]
            if cols_to_drop:
                print(f"[INFO] Dropping non-label columns for {dataset}: {cols_to_drop}")
                labels_df = labels_df.drop(columns=cols_to_drop)
            labels = labels_df.values.astype(np.float32)
            if labels.ndim == 1:
                labels = labels.reshape(-1, 1)
           
            if task_type == "binary" or task_type == "multilabel":
                labels = np.clip(labels, 0.0, 1.0)
        valid_lbls = labels[~np.isnan(labels)]
        print(f"Positive ratio: {(valid_lbls == 1).sum() / max(valid_lbls.size, 1):.4f}")
       
        fps, graphs_raw, coords, adjs, atom_feats, chain_masks = [], [], [], [], [], []
        valid_indices = []
        print("Precomputing cleaned features...")
        for i, smi in enumerate(smiles):
            processed_data = compute_fp_and_descriptors(smi, n_bits=FP_BITS, seed=RUN_SEED)
   
            if processed_data is not None:
                fp_comb, graph_feat_dummy, coord_arr, adj_arr, atom_feat_arr, chain_mask_arr = processed_data
                fps.append(fp_comb)
                graphs_raw.append(graph_feat_dummy) # ¡û dummy 1Î¬
                coords.append(coord_arr)
                adjs.append(adj_arr)
                atom_feats.append(atom_feat_arr)
                chain_masks.append(chain_mask_arr)
                valid_indices.append(i)
            else:
                continue
       
        smiles = [smiles[i] for i in valid_indices]
        labels = labels[valid_indices]
        graphs_raw = np.array(graphs_raw)
        N = len(smiles)
        idx = np.arange(N)
       
        if task_type == "binary":
            stratify = labels.reshape(-1)
            if np.isnan(stratify).any():
                stratify = None
        else:
            stratify = None
           
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            idx, labels, train_size=TRAIN_RATIO, random_state=SPLIT_SEED, stratify=stratify
        )
        val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
       
        stratify_temp = None
        if task_type == "binary":
            stratify_temp = y_temp.reshape(-1)
            if np.isnan(stratify_temp).any():
                stratify_temp = None
               
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_ratio_adjusted, random_state=SPLIT_SEED, stratify=stratify_temp
        )
       
        train_graphs_raw = graphs_raw[train_idx]
        global descriptor_scaler
        descriptor_scaler = StandardScaler()
        descriptor_scaler.fit(train_graphs_raw)
        graphs_transformed = descriptor_scaler.transform(graphs_raw)
        graphs = graphs_transformed.tolist()
       
        train_smiles = [smiles[i] for i in train_idx]
        val_smiles = [smiles[i] for i in val_idx]
        test_smiles = [smiles[i] for i in test_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]
        train_fps = [fps[i] for i in train_idx]
        val_fps = [fps[i] for i in val_idx]
        test_fps = [fps[i] for i in test_idx]
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        test_graphs = [graphs[i] for i in test_idx]
        train_coords = [coords[i] for i in train_idx]
        val_coords = [coords[i] for i in val_idx]
        test_coords = [coords[i] for i in test_idx]
        train_adjs = [adjs[i] for i in train_idx]
        val_adjs = [adjs[i] for i in val_idx]
        test_adjs = [adjs[i] for i in test_idx]
        train_atom_feats = [atom_feats[i] for i in train_idx]
        val_atom_feats = [atom_feats[i] for i in val_idx]
        test_atom_feats = [atom_feats[i] for i in test_idx]
        train_chain_masks = [chain_masks[i] for i in train_idx]
        val_chain_masks = [chain_masks[i] for i in val_idx]
        test_chain_masks = [chain_masks[i] for i in test_idx]
       
        train_ds = SmilesDataset(train_smiles, train_labels, tokenizer, train_fps, train_graphs, train_coords, train_adjs, train_atom_feats, train_chain_masks, augment=AUGMENT_TRAIN)
        val_ds = SmilesDataset(val_smiles, val_labels, tokenizer, val_fps, val_graphs, val_coords, val_adjs, val_atom_feats, val_chain_masks, augment=False)
        test_ds = SmilesDataset(test_smiles, test_labels, tokenizer, test_fps, test_graphs, test_coords, test_adjs, test_atom_feats, test_chain_masks, augment=False)
       
        config = DATASET_SPECIFIC_CONFIG.get(dataset, DEFAULT_CONFIG)
        use_sampler_flag = False
        if task_type == "binary":
            y_train_subset = train_labels.ravel()
            valid_y_train = y_train_subset[~np.isnan(y_train_subset)]
            if len(valid_y_train) > 0:
                pos_ratio = float(np.mean(valid_y_train))
                minority_ratio = min(pos_ratio, 1 - pos_ratio) if 0 < pos_ratio < 1 else 0.5
            else:
                pos_ratio = 0.5
                minority_ratio = 0.5
            use_sampler_flag = bool(config.get("use_sampler", False)) or (AUTO_USE_SAMPLER and (0 < pos_ratio < 1) and (minority_ratio < IMBALANCE_THRESHOLD))
        else:
            use_sampler_flag = bool(config.get("use_sampler", False))
        g = tc.Generator()
        g.manual_seed(RUN_SEED)
        if use_sampler_flag and task_type == "binary":
            y_train_subset = train_labels.ravel()
            valid_y_train = y_train_subset[~np.isnan(y_train_subset)]
            class_counts = np.bincount(valid_y_train.astype(int))
            if len(class_counts) == 2 and class_counts[1] > 0:
                weight_per_class = 1.0 / class_counts
                weights = np.where(~np.isnan(y_train_subset), weight_per_class[np.nan_to_num(y_train_subset).astype(int)], 0)
                sampler_gen = tc.Generator()
                sampler_gen.manual_seed(RUN_SEED)
                sampler = WeightedRandomSampler(
                    weights, num_samples=len(weights), replacement=True,
                    generator=sampler_gen
                )
                tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                       num_workers=NUM_WORKERS, pin_memory=PIN_MEM, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
            else:
                tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=NUM_WORKERS, pin_memory=PIN_MEM, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
        else:
            tr_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=PIN_MEM, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
       
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEM, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEM, collate_fn=collate_fn, worker_init_fn=seed_worker, generator=g)
                                
        num_labels = labels.shape[1] if task_type == "multilabel" else 1
        fp_dim_updated = FP_BITS + 167 + FP_BITS + ATFP_BITS # Ö¸ÎÆ×ÜÎ¬¶È
       
        model = MolModel(
            fp_dim=fp_dim_updated,
            task_type=task_type,
            num_labels=num_labels,
            hidden_dim=HIDDEN_DIM,
            gat_layers=GAT_LAYERS,
            modal_dim=MODAL_DIM,
            heads=HEADS,
            dropout=DROPOUT
        ).to(DEVICE)
       
        # ÓÅ»¯Æ÷ÉèÖÃ£¨±£³ÖÔ­Âß¼­£©
        no_decay = ["bias", "LayerNorm.weight"]
        bert_params = {n: p for n, p in model.bert.named_parameters()}
        other_params = [(n, p) for n, p in model.named_parameters() if not n.startswith("bert.")]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
             'weight_decay': WEIGHT_DECAY, 'lr': LEARNING_RATE},
            {'params': [p for n, p in other_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': LEARNING_RATE},
        ]
       
        # BERT ·Ö²ãÑ§Ï°ÂÊ£¨±£³ÖÔ­Ñù£©
        lr_decay_factor = 0.5
        base_lr = LEARNING_RATE
        num_bert_layers = model.bert.config.num_hidden_layers
        bert_layer_groups = []
        for layer_idx in range(num_bert_layers):
            decay_steps = num_bert_layers - 1 - layer_idx
            current_lr = base_lr * (lr_decay_factor ** decay_steps)
            layer_prefix = f'encoder.layer.{layer_idx}.'
            layer_params_decay = [p for n, p in bert_params.items() if layer_prefix in n and not any(nd in n for nd in no_decay)]
            layer_params_no_decay = [p for n, p in bert_params.items() if layer_prefix in n and any(nd in n for nd in no_decay)]
            if layer_params_decay:
                bert_layer_groups.append({'params': layer_params_decay, 'weight_decay': WEIGHT_DECAY, 'lr': current_lr})
            if layer_params_no_decay:
                bert_layer_groups.append({'params': layer_params_no_decay, 'weight_decay': 0.0, 'lr': current_lr})
        lowest_lr = base_lr * (lr_decay_factor ** num_bert_layers)
        emb_pool_params_decay = [p for n, p in bert_params.items() if ('embeddings.' in n or 'pooler.' in n) and not any(nd in n for nd in no_decay)]
        emb_pool_params_no_decay = [p for n, p in bert_params.items() if ('embeddings.' in n or 'pooler.' in n) and any(nd in n for nd in no_decay)]
        if emb_pool_params_decay:
            bert_layer_groups.append({'params': emb_pool_params_decay, 'weight_decay': WEIGHT_DECAY, 'lr': lowest_lr})
        if emb_pool_params_no_decay:
            bert_layer_groups.append({'params': emb_pool_params_no_decay, 'weight_decay': 0.0, 'lr': lowest_lr})
        optimizer_grouped_parameters.extend(bert_layer_groups)
       
        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        pretrain_optimizer = AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=WEIGHT_DECAY)
       
        pretrain_steps_per_epoch = max(1, len(tr_loader))
        pretrain_total_steps = PRETRAIN_EPOCHS * pretrain_steps_per_epoch
        pretrain_scheduler = get_cosine_schedule_with_warmup(
            pretrain_optimizer,
            num_warmup_steps=int(WARMUP_RATIO * pretrain_total_steps),
            num_training_steps=pretrain_total_steps,
        )
       
        contrastive_loss_fn = ContrastiveLoss()
        scaler = tc.amp.GradScaler('cuda') if tc.cuda.is_available() else None
       
        print(f"[INFO] Starting self-supervised pretraining for {PRETRAIN_EPOCHS} epochs...")
        for pre_epoch in range(PRETRAIN_EPOCHS):
            pretrain_loss = train_one_epoch(
                model, tr_loader, pretrain_optimizer, pretrain_scheduler, None, task_type,
                scaler=scaler, contrastive_loss_fn=contrastive_loss_fn, pretrain_mode=True
            )
            print(f" Pretrain Epoch {pre_epoch+1}/{PRETRAIN_EPOCHS} Loss={pretrain_loss:.4f}")
           
        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        steps_per_epoch = max(1, len(tr_loader))
        total_steps = EPOCHS * steps_per_epoch
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
       
        ytrain_vals = labels[train_idx].reshape(-1)
        loss_fn = get_loss_fn_for_training(task_type, ytrain_vals, dataset_name=dataset)
       
        best_metric = -9999.0
        best_epoch = -1
        patience_count = 0
        dataset_name_clean = os.path.splitext(os.path.basename(dataset))[0]
        best_path = os.path.join(checkpoint_dir, f"{dataset_name_clean}_seed{RUN_SEED}.pt")
       
        for epoch in range(EPOCHS):
            t0 = time.time()
            train_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, loss_fn, task_type, scaler=scaler, contrastive_loss_fn=contrastive_loss_fn, epoch=epoch)
            val_metrics = evaluate(model, val_loader, task_type, dataset_name=dataset)
            t1 = time.time()
            monitor = val_metrics.get("AUC", -val_metrics.get("RMSE", -9999.0))
            scheduler.step(monitor)
           
            if not (isinstance(monitor, float) and np.isnan(monitor)) and monitor > best_metric:
                best_metric = monitor
                best_epoch = epoch
                patience_count = 0
                try:
                    tc.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "dataset": dataset
                    }, best_path)
                    print(f" [SAVED] New best model at epoch {epoch+1}, AUC={monitor:.4f} ¡ú {best_path}")
                except Exception as e:
                    print(f"[WARN] Failed to save checkpoint: {e}")
            else:
                patience_count += 1
            print(f" Epoch {epoch+1}/{EPOCHS} TrainLoss={train_loss:.4f} ValMetrics={val_metrics} Time={t1-t0:.1f}s BestEpoch={best_epoch+1}")
            if patience_count >= PATIENCE:
                print(" Early stopping")
                break
               
        if os.path.exists(best_path):
            try:
                ckpt = tc.load(best_path, map_location=DEVICE, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"[LOAD] Best model loaded from epoch {ckpt['epoch']+1}")
            except Exception as e:
                print(f"[WARN] Failed to load checkpoint: {e}")
               
        test_metrics = evaluate(model, test_loader, task_type, dataset_name=dataset)
        print(f"[{dataset}] Test metrics: {test_metrics}")
        results[dataset] = {"val_best_metric": best_metric, "test_metrics": test_metrics}
   
    return results


if __name__ == "__main__":
    run()