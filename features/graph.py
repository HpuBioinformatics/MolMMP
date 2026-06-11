import os
import numpy as np
import torch as tc
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs,Descriptors

def get_atom_features(mol):
    
    if mol is None:
        return np.zeros((0, 10), dtype=np.float32)
    atom_features = []
    for atom in mol.GetAtoms():
        feat = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetHybridization().real,
            atom.GetTotalNumHs(includeNeighbors=True),
            atom.GetImplicitValence(),
            atom.GetIsotope(),
            atom.IsInRing(),
            atom.GetChiralTag().real,
        ]
        atom_features.append(feat)
    return np.array(atom_features, dtype=np.float32)


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


def compute_fp_and_descriptors(smiles, n_bits=1024, seed=0):
  
    log_file = "./result/invalid_smiles.txt"
    os.makedirs("./result", exist_ok=True)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol3d = Chem.AddHs(mol)
        etkdg = AllChem.ETKDGv3()
        etkdg.randomSeed = int(seed)

        embed_success = False
        for attempt in range(3):
            etkdg.randomSeed = int(seed + attempt)
            if AllChem.EmbedMolecule(mol3d, etkdg) == 0:
                embed_success = True
                break

        if not embed_success:
            with open(log_file, 'a') as f:
                f.write(f"EmbedMolecule failed after 3 attempts (fallback to 2D): {smiles}\n")
            mol_cleaned = Chem.RemoveHs(mol)
            AllChem.Compute2DCoords(mol_cleaned)
            conf = mol_cleaned.GetConformer()
        else:
            try:
                AllChem.MMFFOptimizeMolecule(mol3d, maxIters=500)
            except Exception as mmff_err:
                try:
                    AllChem.UFFOptimizeMolecule(mol3d, maxIters=500)
                    with open(log_file, 'a') as f:
                        f.write(f"MMFF failed, used UFF: {smiles}\n")
                except Exception as uff_err:
                    with open(log_file, 'a') as f:
                        f.write(f"MMFF & UFF failed (fallback to 2D): {smiles}\n")
                    mol_cleaned = Chem.RemoveHs(mol3d)
                    AllChem.Compute2DCoords(mol_cleaned)
                    conf = mol_cleaned.GetConformer()
            else:
                mol_cleaned = Chem.RemoveHs(mol3d)
                conf = mol_cleaned.GetConformer()

        n_atoms = mol_cleaned.GetNumAtoms()


        coords = np.zeros((n_atoms, 3), dtype=np.float32)
        for j in range(n_atoms):
            pos = conf.GetAtomPosition(j)
            coords[j] = [pos.x, pos.y, pos.z]
        if n_atoms > 0:
            coords = coords - np.mean(coords, axis=0)

      
        fp_comb = compute_fingerprint(mol_cleaned, n_bits)

        
        atom_feats = get_atom_features(mol_cleaned)
        adj = np.zeros((n_atoms, n_atoms), dtype=np.float32)
        for b in mol_cleaned.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            adj[i, j] = adj[j, i] = 1.0

        chain_mask = np.zeros(n_atoms, dtype=np.float32)
        if n_atoms > 1:
            try:
                path = Chem.GetShortestPath(mol_cleaned, 0, n_atoms - 1)
                chain_mask[path] = 1.0
            except:
                pass

        
        graph_feat_dummy = np.zeros(1, dtype=np.float32)

        fp_comb = np.nan_to_num(fp_comb, nan=0.0, posinf=0.0, neginf=0.0)
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
        adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        atom_feats = np.nan_to_num(atom_feats, nan=0.0, posinf=0.0, neginf=0.0)
        chain_mask = np.nan_to_num(chain_mask, nan=0.0, posinf=0.0, neginf=0.0)

        return fp_comb, graph_feat_dummy, coords, adj, atom_feats, chain_mask

    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Runtime error: {smiles} | {str(e)}\n")
        return None