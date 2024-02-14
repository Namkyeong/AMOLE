import sys
sys.path.append('.')

import os
from itertools import repeat
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import mol_to_graph_data_obj_simple


class DrugBank_Datasets_Graph_ATC(InMemoryDataset):
    def __init__(
        self, root, file_name, processed_dir_prefix, neg_sample_size, prompt_template="{}.",
        transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.file_name = file_name
        self.processed_dir_prefix = processed_dir_prefix
        self.smiles_text_file_name = "SMILES.csv"
        self.prompt_template = prompt_template

        super(DrugBank_Datasets_Graph_ATC, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        self.SMILES_list = df["smiles"].tolist()
        self.ATC_code_list = df["ATC_code"].tolist()
        ATC_label_list = df["ATC_label"].tolist() # This is for raw TAC label
        self.ATC_label_list = [self.prompt_template.format(x) for x in ATC_label_list]

        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", file_name)
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list

        assert len(self.SMILES_list) == len(self.neg_index_list) == len(self.ATC_code_list) == len(self.ATC_label_list)
        return

    def get_graph(self, index):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[index], slices[index + 1])
            data[key] = item[s]
        return data

    def get(self, index):
        text = self.ATC_label_list[index]
        data = self.get_graph(index)
        neg_index_list = np.random.choice(self.neg_index_list[index], self.neg_sample_size)
        # neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.ATC_label_list[idx] for idx in neg_index_list]
        # neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get_graph(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, "processed", "molecule_{}".format(self.processed_dir_prefix))

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        SMILES2ATC_txt_file = os.path.join(self.root, "raw", self.file_name)
        
        f = open(SMILES2ATC_txt_file, 'r')
        data_list, SMILES_list, ATC_code_list, ATC_label_list = [], [], [], []
        for line_idx, line in enumerate(f.readlines()):
            line = line.strip().split("\t")
            SMILES = line[0]
            ATC_code = line[1]
            ATC_label = line[2]
            rdkit_mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(rdkit_mol)
            data.id = torch.tensor([line_idx])

            data_list.append(data)
            SMILES_list.append(SMILES)
            ATC_code_list.append(ATC_code)
            ATC_label_list.append(ATC_label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"smiles": SMILES_list, "ATC_code": ATC_code_list, "ATC_label": ATC_label_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        return

    def __len__(self):
        return len(self.SMILES_list)


if __name__ == "__main__":
    
    DATA_PATH = "./data/Drugbank"
    
    # Test Loader
    T_max = 9
    prompt_template = "This molecule is for {}."
    full_file_name = "SMILES_ATC_5_full.txt"
    processed_dir_prefix = "ATC_full_5"
    
    batch_size = 32
    num_workers = 6
    
    dataset_class = DrugBank_Datasets_Graph_ATC
    dataloader_class = pyg_DataLoader
    
    dataset = dataset_class(DATA_PATH, full_file_name, processed_dir_prefix, neg_sample_size=T_max, prompt_template=prompt_template)
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")