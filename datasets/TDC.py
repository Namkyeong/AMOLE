import sys
sys.path.append('.')

import os
from itertools import repeat
import pandas as pd
import json
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import mol_to_graph_data_obj_simple


class TDC_Datasets_Graph(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.download_dir = os.path.join(self.root, "TDC" ,"{}".format(self.name), "raw")
        
        super(TDC_Datasets_Graph, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        return

    @property
    def processed_dir(self):
        return os.path.join(self.root, "TDC" ,"{}".format(self.name), "processed")

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        
        data_list = []

        if self.name == "HIA":
            from tdc.single_pred import ADME
            data = ADME(name = "HIA_Hou", path = self.download_dir)
            df = data.get_data()

        if self.name == "Bioavailability":
            from tdc.single_pred import ADME
            data = ADME(name = "Bioavailability_Ma", path = self.download_dir)
            df = data.get_data()

        if self.name == "Pgp_Inhibition":
            from tdc.single_pred import ADME
            data = ADME(name = "Pgp_Broccatelli", path = self.download_dir)
            df = data.get_data()

        if self.name == "BBB":
            from tdc.single_pred import ADME
            data = ADME(name = "BBB_Martins", path = self.download_dir)
            df = data.get_data()
                
        if self.name == "hERG":
            from tdc.single_pred import Tox
            data = Tox(name = "hERG", path = self.download_dir)
            df = data.get_data()

        if self.name == "DILI":
            from tdc.single_pred import Tox
            data = Tox(name = 'DILI', path = self.download_dir)
            df = data.get_data()

        if self.name == "HIV":
            from tdc.single_pred import HTS
            data = HTS(name = 'HIV', path = self.download_dir)
            df = data.get_data()
        
        if self.name == "VDR":
            temp_dir = os.path.join(self.root, "TDC", "{}".format(self.name), "raw", "vdr_vs_data.csv")
            df = pd.read_csv(temp_dir)
            df = df.rename(columns={"SMILES": "Drug", "label": "Y"})

        for i in range(len(df)):
            SMILES = df["Drug"][i]
            rdkit_mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(rdkit_mol)
            data["label"] = torch.tensor(df["Y"][i])
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))

        return

    def get(self, index):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[index], slices[index + 1])
            data[key] = item[s]
        return data


if __name__ == "__main__":
    
    DATA_PATH = "./data/"
    name = "HIA"
    batch_size = 32
    num_workers = 6
    
    dataset = TDC_Datasets_Graph(DATA_PATH, name)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")