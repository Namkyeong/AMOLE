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


class DrugBank_Datasets_Graph_retrieval(InMemoryDataset):
    def __init__(
        self, root, train_mode, neg_sample_size, processed_dir_prefix, template="raw/SMILES_description_{}.txt",
        transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.processed_dir_prefix = processed_dir_prefix
        self.template = template
        self.train_mode = train_mode
        self.smiles_text_file_name = "SMILES.csv"

        super(DrugBank_Datasets_Graph_retrieval, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Data: {}'.format(self.data))

        df = pd.read_csv(os.path.join(self.processed_dir, self.smiles_text_file_name))
        print(df.columns)
        self.text_list = df["text"].tolist()

        # sampling
        self.neg_sample_size = neg_sample_size
        negative_sampled_index_file = os.path.join(self.root, "index", template.format(train_mode))
        print("Loading negative samples from {}".format(negative_sampled_index_file))
        f = open(negative_sampled_index_file, 'r')
        neg_index_list = []
        for line in f.readlines():
            line = line.strip().split(",")
            line = [int(x) for x in line]
            neg_index_list.append(line)
        self.neg_index_list = neg_index_list
        
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
        text = self.text_list[index]
        data = self.get_graph(index)
        neg_index_list = np.random.choice(self.neg_index_list[index], self.neg_sample_size)
        # neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_text = [self.text_list[idx] for idx in neg_index_list]
        # neg_index_list = self.neg_index_list[index][:self.neg_sample_size]
        neg_data = [self.get_graph(idx) for idx in neg_index_list]
        return text, data, neg_text, neg_data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', '{}_{}'.format(self.processed_dir_prefix, self.train_mode))

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list, SMILES_list, text_list = [], [], []
        SMILES2description_file = os.path.join(self.root, 'raw', self.template.format(self.train_mode))
        f = open(SMILES2description_file, 'r')

        for line_id, line in enumerate(tqdm(f.readlines())):
            line = line.strip().split("\t", 1)
            SMILES = line[0]
            text = line[1]

            rdkit_mol = AllChem.MolFromSmiles(SMILES)
            data = mol_to_graph_data_obj_simple(rdkit_mol)
            data.id = torch.tensor([line_id])

            data_list.append(data)
            SMILES_list.append(SMILES)
            text_list.append(text)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        df = pd.DataFrame(
            {"text": text_list, "smiles": SMILES_list},
        )
        saver_path = os.path.join(self.processed_dir, self.smiles_text_file_name)
        print("saving to {}".format(saver_path))
        df.to_csv(saver_path, index=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("saving to {}".format(self.processed_paths[0]))
        print()
        return

    def __len__(self):
        return len(self.text_list)


if __name__ == "__main__":
    
    DATA_PATH = "./data/Drugbank"
    
    # Test Loader
    T_max = 9
    processed_dir_prefix = "molecule_pharmacodynamics_removed_PubChem"
    template = "SMILES_pharmacodynamics_removed_from_PubChem_{}.txt"
    
    batch_size = 32
    num_workers = 6
    
    dataset = DrugBank_Datasets_Graph_retrieval(DATA_PATH, 'full', neg_sample_size=T_max, processed_dir_prefix=processed_dir_prefix, template=template)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")