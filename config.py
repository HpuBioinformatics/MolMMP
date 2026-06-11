# -*- coding: utf-8 -*-
import torch as tc
import os

# ---------------- Config ----------------
BERT_PATH = "/your/ChemBERTa-77M-MLM/"
DATA_ROOT = "/your/Drug Safety Assessment/"
MOLECULENET_ROOT = "/your/MoleculeNet/" 

DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")
FP_BITS = 1024
ATFP_BITS = 200

# ---------------- run-seed ----------------
RUN_SEED = 43
SPLIT_SEED = 42
NUM_WORKERS = 4
PIN_MEM = True if tc.cuda.is_available() else False
COORD_JITTER_SIGMA = 0.05
SMILES_AUGMENT_NUM = 3
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
EPOCHS = 100 
PRETRAIN_EPOCHS = 10
PRETRAIN_LR = 1e-5
LEARNING_RATE = 2.5e-4 
BATCH_SIZE = 64 
WEIGHT_DECAY = 4.95e-4
HEADS = 8
HIDDEN_DIM = 408
GAT_LAYERS = 4
DROPOUT = 0.5
MODAL_DIM = 256
GAMMA = 2.0 
ALPHA = 0.5 
BERT_UNFREEZE_LAYERS = 4
WARMUP_RATIO = 0.6 
CLIP_NORM = 1.0
PATIENCE = 20 
AUGMENT_TRAIN = True
DEBUG = False
CONTRASTIVE_TEMP = 0.07
CONTRASTIVE_WEIGHT = 0.1
CONTRAST_LAMBDA = 0.05 
USE_CONTRASTIVE_REG = True
THREE_D_TRANSFORMER_LAYERS = 4
THREE_D_EMBED_DIM = 128
MAX_ATOMS = 512

# ---------------- Dataset-Specific ----------------
DATASET_SPECIFIC_CONFIG = {
    "CTMTL/CTMTL.csv": {
        "focal_alpha": 0.75,
        "focal_gamma": 3.5,
        "binary_threshold": 0.5,
        "use_pos_weight": True,
        "use_sampler": True,
    }
}
DEFAULT_CONFIG = {
    "focal_alpha": ALPHA,
    "focal_gamma": GAMMA,
    "binary_threshold": 0.5,
    "use_pos_weight": False,
    "use_sampler": False,
}
AUTO_USE_SAMPLER = True
AUTO_USE_POS_WEIGHT = True
IMBALANCE_THRESHOLD = 0.35

DATASETS = {
    "DILI.csv": "binary",
    "Acute Oral Toxicity.csv": "binary",
    "tox21.csv": "multilabel",
    "clintox.csv": "multilabel",
    "hERG.csv": "binary",
    "Mutagenicity.csv": "binary",
}