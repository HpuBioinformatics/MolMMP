# -*- coding: utf-8 -*-
import numpy as np
import torch as tc
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from config import COORD_JITTER_SIGMA, AUGMENT_TRAIN

class SmilesDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer, fps, graphs, coords, adjs, atom_feats, chain_masks, augment=False):
        self.smiles = smiles
        self.labels = labels
        self.tokenizer = tokenizer
        self.fps = fps
        self.graphs = graphs
        self.coords = coords
        self.adjs = adjs
        self.atom_feats = atom_feats
        self.chain_masks = chain_masks
        self.augment = augment

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.augment:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smi = Chem.MolToSmiles(mol, isomericSmiles=True, doRandom=True, canonical=True)
        token = self.tokenizer(smi, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        coords = self.coords[idx]
        if self.augment and coords.shape[0] > 0:
            coords = coords.copy()
            coords += np.random.normal(0, COORD_JITTER_SIGMA, coords.shape).astype(np.float32)
        return {
            "input_ids": token["input_ids"].squeeze(0),
            "attention_mask": token["attention_mask"].squeeze(0),
            "fp": self.fps[idx],
            "graph_feats": self.graphs[idx],
            "coords": coords,
            "adj": self.adjs[idx],
            "atom_feats": self.atom_feats[idx],
            "chain_mask": self.chain_masks[idx],
            "labels": self.labels[idx]
        }

def collate_fn(batch):
    input_ids = tc.stack([item["input_ids"] for item in batch])
    attention_mask = tc.stack([item["attention_mask"] for item in batch])
    fp = tc.tensor(np.stack([item["fp"] for item in batch], axis=0), dtype=tc.float32)
    graph_feats = tc.tensor(np.stack([item["graph_feats"] for item in batch], axis=0), dtype=tc.float32)
    labels = tc.tensor(np.stack([item["labels"] for item in batch], axis=0), dtype=tc.float32)
    coords_list = [tc.tensor(item["coords"], dtype=tc.float32) for item in batch]
    adj_list = [tc.tensor(item["adj"], dtype=tc.float32) for item in batch]
    atom_feats_list = [tc.tensor(item["atom_feats"], dtype=tc.float32) for item in batch]
    chain_masks_list = [tc.tensor(item["chain_mask"], dtype=tc.float32) for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "fp": fp,
        "graph_feats": graph_feats,
        "labels": labels,
        "coords_list": coords_list,
        "adj_list": adj_list,
        "atom_feats_list": atom_feats_list,
        "chain_masks_list": chain_masks_list
    }