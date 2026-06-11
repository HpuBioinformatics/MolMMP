# -*- coding: utf-8 -*-
import os, random, time, warnings
import numpy as np
import torch as tc
from sklearn.preprocessing import StandardScaler

descriptor_scaler = StandardScaler()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)
        tc.backends.cudnn.deterministic = True
        tc.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    worker_seed = tc.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def torch_safe_nan_to_num(x, device, dtype):
    return tc.where(tc.isfinite(x), x, tc.zeros_like(x, dtype=dtype, device=device))