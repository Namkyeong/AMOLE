import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import QA_Datasets_Graph_retrieval
from torch_geometric.loader import DataLoader as pyg_DataLoader

# For Language Models
from transformers import AutoModel, AutoTokenizer
from utils.bert import prepare_text_tokens

# For Graph Neural Networks
from layers import GNN, GNN_graphpred


def do_CL_eval(X, Y, label):
    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1) # B, 1, d

    Y = Y.transpose(0, 1)  # B, T, d
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze()  # B*T
    B = X.size()[0]

    pred = logits.argmax(dim=1, keepdim=False)
    

    return pred, label


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
def eval_epoch(dataloader):

    text_model.eval()
    molecule_model.eval()
    text2latent.eval()
    mol2latent.eval()

    preds, labels = list(), list()

    for batch in tqdm(dataloader):
        
        molecule_data = batch[0]
        text = batch[1]
        label = batch[2]

        text_repr = [get_text_repr(text[idx]) for idx in range(5)]
        text_repr = torch.stack(text_repr)
        molecule_repr = get_molecule_repr(molecule_data.to(device))

        pred, label = do_CL_eval(molecule_repr, text_repr, label.to(device))
        
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())
    
    preds = torch.hstack(preds)
    labels = torch.hstack(labels)
    acc = preds.eq(labels).float().mean().item()
    
    return acc, preds, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--text_type", type=str, default="SciBERT", choices=["SciBERT", "BioBERT"])

    parser.add_argument("--dataspace_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="description", choices=["description", "pharmacodynamics"])
    parser.add_argument("--input_model_dir", type=str, default="./model_checkpoints")
    parser.add_argument("--input_model_config", type=str, 
                        default="AMOLE.pth")

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
    print("arguments\t", args)

    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = "2"

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    

    ##### prepare text model #####
    if args.text_type == "SciBERT":
        pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'PubChemSTM', 'pretrained_SciBERT')
        text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
        # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
        text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
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
    
    dataset = QA_Datasets_Graph_retrieval(args.dataspace_path, args.dataset)
    dataloader_class = pyg_DataLoader
    dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_acc, preds, labels = eval_epoch(dataloader)
    
    print("arguments\t", args)
    print(test_acc)

     # Write experimental results
    WRITE_PATH = "results_QA/"
    os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
    f = open("results_QA/{}.txt".format(args.dataset), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write("{}".format(args.input_model_config))
    f.write("\n")
    f.write("Acc: {}".format(test_acc))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()