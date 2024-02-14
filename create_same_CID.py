import os
import pandas as pd
import numpy as np
import torch

CID_text_file = pd.read_csv("./data/PubChemSTM/processed/CID_text_list.csv")

dict = {}
for i in range(len(CID_text_file)):
    CID = CID_text_file["CID"][i]
    idx_list = np.where(CID_text_file["CID"] == CID)
    idx_list = np.delete(idx_list[0], np.where(idx_list[0] == i)[0][0])
    dict[i] = idx_list

SAVE_PATH = "./data/PubChemSTM/processed/"
os.makedirs(SAVE_PATH, exist_ok=True) # Create directory if it does not exist
torch.save(dict, "./data/PubChemSTM/processed/same_CID.pt")