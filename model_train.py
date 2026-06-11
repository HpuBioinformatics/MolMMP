# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.deprecation')

import os
os.environ["RDKIT_LOGLEVEL"] = "ERROR"
os.environ["PYTHONWARNINGS"] = "ignore"
import os
from config import RUN_SEED
from utils import set_seed
from train import run

if __name__ == "__main__":
    set_seed(RUN_SEED)
    print("=== Starting Molecular Toxicity Prediction Training ===")
    results = run()
    print("\n=== All Training Completed ===")
    print(results)