import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import torch
from rdkit.Chem import AllChem
from rdkit import DataStructs

cid2smiles = pd.read_csv("./data/PubChemSTM/raw/CID2SMILES.csv")

fp_lists = []
for index, smiles in enumerate(tqdm(cid2smiles.SMILES)):
    mol = AllChem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    array = np.zeros((0, ), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    fp = torch.tensor(array).reshape(1, -1)
    fp_lists.append(fp)

fps_stm = torch.vstack(fp_lists)

# Create similarity list among the molecules in PubChemSTM dataset
similarity_lists = list()
index_lists = list()

for index, mol in enumerate(tqdm(fps_stm)):
    intersection = torch.logical_and(fps_stm[index], fps_stm)
    union = torch.logical_or(fps_stm[index], fps_stm)
    tanimoto_sim = intersection.sum(dim = 1) / union.sum(dim = 1)
    values, indices = tanimoto_sim.topk(100)
    
    similarity_lists.append(values)
    index_lists.append(indices)

similarities = torch.vstack(similarity_lists)
similar_idxs = torch.vstack(index_lists)

similarities_CID = dict()
for i, cid in enumerate(cid2smiles.CID):
    similarities_CID[cid] = np.asarray(similar_idxs[i])

SAVE_PATH = "./data/PubChemSTM/processed/"
os.makedirs(SAVE_PATH, exist_ok=True) # Create directory if it does not exist
torch.save(similarities_CID, "./data/PubChemSTM/processed/similarities_CID.pt")