import sys
sys.path.append('.')

import os
from itertools import repeat
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import pickle
import random

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
RDLogger.DisableLog('rdApp.*')

from utils.chem import mol_to_graph_data_obj_simple


class TanimotoSTM_Datasets_Graph(InMemoryDataset):
    def __init__(self, root, aug = 0.0, num_cand = 30, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        # only for `process` function
        self.SDF_file_path = os.path.join(self.root, "raw/molecules.sdf")
        self.CID2text_file = os.path.join(self.root, "raw/CID2text.json")
        # `process` result file
        self.CID_text_file_path = os.path.join(self.root, "processed/CID_text_list.csv")
        # `similarity` result file
        self.similarity_file_path = os.path.join(self.root, "processed/similarities_CID.pt")

        self.same_cid_path = os.path.join(self.root, "processed/same_CID.pt")

        super(TanimotoSTM_Datasets_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        self.p_aug = aug
        self.num_cand = num_cand

        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        suppl = Chem.SDMolSupplier(self.SDF_file_path)

        CID2graph = {}
        for mol in tqdm(suppl):
            CID = mol.GetProp("PUBCHEM_COMPOUND_CID")
            CID = int(CID)
            graph = mol_to_graph_data_obj_simple(mol)

            # Create Fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            array = np.zeros((0, ), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, array)
            fp = torch.tensor(array).reshape(1, -1)
            graph.fp = fp
            
            CID2graph[CID] = graph
        print("CID2graph", len(CID2graph))
        
        with open(self.CID2text_file, "r") as f:
            CID2text_data = json.load(f)
        print("CID2data", len(CID2text_data))
            
        CID_list, graph_list, text_list = [], [], []
        for CID, value_list in CID2text_data.items():
            CID = int(CID)
            if CID not in CID2graph:
                print("CID {} missing".format(CID))
                continue
            graph = CID2graph[CID]
            for value in value_list:
                text_list.append(value)
                CID_list.append(CID)
                graph_list.append(graph)

        CID_text_df = pd.DataFrame({"CID": CID_list, "text": text_list})
        CID_text_df.to_csv(self.CID_text_file_path, index=None)

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices), self.processed_paths[0])

        return
    
    def load_Graph_CID_and_text(self):
        self.graphs, self.slices = torch.load(self.processed_paths[0])

        CID_text_df = pd.read_csv(self.CID_text_file_path)
        self.CID_list = CID_text_df["CID"].tolist()
        self.text_list = CID_text_df["text"].tolist()

        # Load similar molecules
        self.similarity = torch.load(self.similarity_file_path)

        self.same_CID = torch.load(self.same_cid_path)

        self.CID_key_list = list()
        for i in range(len(self.same_CID)):
            if len(self.same_CID[i]) > 0:
                self.CID_key_list.append(i)

        return


    def get(self, idx):
        text = self.text_list[idx]

        aux_idx = np.random.choice(self.CID_key_list)
        aux_text = self.text_list[aux_idx]
        sameCIDidx = np.random.choice(self.same_CID[aux_idx])
        new_text = self.text_list[sameCIDidx]
        aux_text += " [SEP] " + new_text

        CID = self.CID_list[idx]
        similar_CIDs = self.similarity[CID][:self.num_cand]

        similar_CID = np.random.choice(similar_CIDs)
        similar_index = self.CID_list.index(similar_CID)

        # Augmentation percent
        do_aug = np.random.binomial(1, self.p_aug)

        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            if do_aug == 1:
                s[data.__cat_dim__(key, item)] = slice(slices[similar_index], slices[similar_index + 1])
            elif do_aug == 0:
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                raise Exception
            data[key] = item[s]
        
        original_data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            original_data[key] = item[s]

        return text, original_data, data, aux_text

    def __len__(self):
        return len(self.text_list)



if __name__ == "__main__":
    
    DATA_PATH = "./data/PubChemSTM"
    batch_size = 45
    num_workers = 6
    
    dataset = TanimotoSTM_Datasets_Graph(DATA_PATH)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")