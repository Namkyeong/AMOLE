import os
import argparse
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import TDC_Datasets_Graph
from torch_geometric.loader import DataLoader as pyg_DataLoader

# For Language Models
from transformers import AutoModel, AutoTokenizer
from utils.bert import prepare_text_tokens

# For Graph Neural Networks
from layers import GNN, GNN_graphpred

def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_text_repr(text):
    text_tokens_ids, text_masks = prepare_text_tokens(
        device=device, description=text, tokenizer=text_tokenizer, max_seq_len=args.max_seq_len)
    text_output = text_model(input_ids=text_tokens_ids, attention_mask=text_masks)
    text_repr = text_output["pooler_output"]
    text_repr = text2latent(text_repr)
    return text_repr


def get_molecule_repr(molecule):

    molecule_output, _ = molecule_model(molecule.to(device))
    molecule_repr = mol2latent(molecule_output)

    return molecule_repr


@torch.no_grad()
def molecule_screening(dataloader, prompt, k):

    text_model.eval()
    molecule_model.eval()
    text2latent.eval()
    mol2latent.eval()

    text_repr = get_text_repr([prompt]).detach().cpu()
    molecule_reprs, labels = list(), list()

    for molecule_data in tqdm(dataloader):
        
        molecule_repr = get_molecule_repr(molecule_data.to(device))
        molecule_reprs.append(molecule_repr.detach().cpu())
        labels.append(molecule_data.label.detach().cpu())

    molecule_reprs = torch.vstack(molecule_reprs)
    labels = torch.hstack(labels)

    text_repr = F.normalize(text_repr, dim = -1).to(device)
    molecule_reprs = F.normalize(molecule_reprs, dim = -1).to(device)

    similarities = torch.matmul(text_repr, molecule_reprs.T).detach().cpu()    
    _, indices = similarities.reshape(-1).topk(k)
    hit_rate = labels[indices].float().mean()

    return hit_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--text_type", type=str, default="SciBERT", choices=["SciBERT", "BioBERT", "SentenceBERT"])

    parser.add_argument("--dataspace_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="HIA")
    parser.add_argument("--input_model_dir", type=str, default="./model_checkpoints")
    parser.add_argument("--input_model_config", type=str, 
                        default="AMOLE.pth")

    parser.add_argument("--prompt", type=str, default="Human intestinal absorption (HIA)")
    parser.add_argument("--topk", type=int, default=100)

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=128)

    ########## for BERT model ##########
    parser.add_argument("--max_seq_len", type=int, default=512)

    ########## for 2D GNN ##########
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    args = parser.parse_args()
    print("Task\t", args.dataset)
    print("Prompt\t", args.prompt)
    print("Model\t", args.input_model_config)    
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    ##### prepare text model #####
    if args.text_type == "SciBERT":
        pretrained_SciBERT_folder = os.path.join("./data", 'PubChemSTM' ,'pretrained_SciBERT')
        text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
        # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
        text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
        text_dim = 768
    elif args.text_type == "SentenceBERT":
        pretrained_SentenceBERT_folder = os.path.join("./data", 'PubChemSTM' ,'pretrained_SentenceBERT')
        text_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1', cache_dir=pretrained_SentenceBERT_folder)
        # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
        text_model = AutoModel.from_pretrained('sentence-transformers/multi-qa-mpnet-base-dot-v1', cache_dir=pretrained_SentenceBERT_folder).to(device)
        text_dim = 768
    else:
        raise Exception
    
    ##### prepare molecule model #####
    molecule_node_model = GNN(
        num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
        JK=args.JK, drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type)
    molecule_model = GNN_graphpred(
        num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
        num_tasks=1, molecule_node_model=molecule_node_model) 
    molecule_dim = args.gnn_emb_dim

    text2latent = nn.Linear(text_dim, args.SSL_emb_dim)
    mol2latent = nn.Linear(molecule_dim, args.SSL_emb_dim)

    # Load pretrained checkpoints
    input_model_path_file = args.input_model_config

    input_model_path = os.path.join(args.input_model_dir, "text", input_model_path_file)
    state_dict = torch.load(input_model_path, map_location='cpu')
    text_model.load_state_dict(state_dict)

    input_model_path = os.path.join(args.input_model_dir, "molecule", input_model_path_file)
    state_dict = torch.load(input_model_path, map_location='cpu')
    molecule_model.load_state_dict(state_dict)

    input_model_path = os.path.join(args.input_model_dir, "text2latent", input_model_path_file)
    state_dict = torch.load(input_model_path, map_location='cpu')
    text2latent.load_state_dict(state_dict)

    input_model_path = os.path.join(args.input_model_dir, "mol2latent", input_model_path_file)
    state_dict = torch.load(input_model_path, map_location='cpu')
    mol2latent.load_state_dict(state_dict)

    text_model = text_model.to(device)
    molecule_model = molecule_model.to(device)
    text2latent = text2latent.to(device)
    mol2latent = mol2latent.to(device)

    dataset = TDC_Datasets_Graph(args.dataspace_path, args.dataset)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    hit_rate = molecule_screening(dataloader, args.prompt, args.topk)

    # Write experimental results
    WRITE_PATH = "results_vs/"
    os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
    f = open("results_vs/{}.txt".format(args.dataset), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write("Prompt: {}".format(args.prompt))
    f.write("\n")
    f.write("Model: {}".format(args.input_model_config))
    f.write("\n")
    f.write("Hit Rate: {}".format(hit_rate))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()