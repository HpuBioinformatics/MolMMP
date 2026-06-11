import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs

def compute_fingerprint(mol, n_bits=1024):
    
    fp_ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
    arr_ecfp = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_ecfp, arr_ecfp)

    fp_maccs = MACCSkeys.GenMACCSKeys(mol)
    arr_maccs = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_maccs, arr_maccs)

    fp_rdkit = Chem.RDKFingerprint(mol, minPath=1, maxPath=5, fpSize=n_bits)
    arr_rdkit = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_rdkit, arr_rdkit)

    fp_atfp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=200)
    arr_atfp = np.zeros((200,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp_atfp, arr_atfp)

    return np.concatenate([arr_ecfp, arr_maccs, arr_rdkit, arr_atfp]).astype(np.float32)