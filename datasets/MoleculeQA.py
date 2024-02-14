import sys
sys.path.append('.')

import os
from itertools import repeat
import pandas as pd
import numpy as np
from tqdm import tqdm

import json

import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader as pyg_DataLoader

from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils.chem import mol_to_graph_data_obj_simple

# This is for validated QA Datasets
class QA_Datasets_Graph_retrieval(InMemoryDataset):
    def __init__(
        self, root, name, transform=None, pre_transform=None, pre_filter=None, empty=False
    ):
        self.root = root
        self.name = name
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform

        self.qa_csv_file_path = os.path.join(self.root, "MoleculeQA/{}/raw".format(name))

        super(QA_Datasets_Graph_retrieval, self).__init__(root, transform, pre_transform, pre_filter)

        self.load_Graph_CID_and_text()
        return

    
    def load_Graph_CID_and_text(self):
        self.graphs, self.slices, self.options, self.labels_list = torch.load(self.processed_paths[0])

        return

    def get_graph(self, index):
        data = Data()
        for key in self.graphs.keys:
            item, slices = self.graphs[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[index], slices[index + 1])
            data[key] = item[s]
        return data

    def get(self, index):
        
        data = self.get_graph(index)
        options = self.options[index]
        label = self.labels_list[index]
        
        return data, options, label

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'MoleculeQA', self.name, 'processed')

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):

        # Read QA Files
        df = pd.read_csv(self.qa_csv_file_path + "/{}_validated.csv".format(self.name))

        # Filter the data (Validation Step)
        df = df[df["correct_option"] == df["validated_option"]]
        df = df.reset_index()

        SMILES_list = df["smiles"]
        unique_SMILES_list = np.unique(SMILES_list)

        SMILES2Graph = {}
        for SMILES in tqdm(unique_SMILES_list):
            mol = AllChem.MolFromSmiles(SMILES)
            graph = mol_to_graph_data_obj_simple(mol)
            SMILES2Graph[SMILES] = graph
        print("SMILES2graph", len(SMILES2Graph))
        
        SMILES_list, graph_list, options_list, labels_list = list(), list(), list(), list()
        
        for i in tqdm(range(len(df))):
            SMILES = df["smiles"][i]
            graph = SMILES2Graph[SMILES]
            options = eval(df["options"][i])
            label = int(df["correct_option"][i] - 1)

            SMILES_list.append(SMILES)
            graph_list.append(graph)
            options_list.append(options)
            labels_list.append(label)
        
        print("Total Converted Data: {}".format(len(SMILES_list)))

        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(graph) for graph in graph_list]

        # Save graphs
        graphs, slices = self.collate(graph_list)
        torch.save((graphs, slices, options_list, labels_list), self.processed_paths[0])
        return

    def __len__(self):
        return len(self.labels_list)


if __name__ == "__main__":
    
    DATA_PATH = "./data"
    batch_size = 32
    num_workers = 6
    
    dataset = QA_Datasets_Graph_retrieval(DATA_PATH, "description")
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_graph_batch = next(iter(dataloader))
    
    print("Hi")