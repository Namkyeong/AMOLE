import os
import argparse
import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import DrugBank_Datasets_Graph_retrieval, DrugBank_Datasets_Graph_ATC
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

def do_CL_eval(X, Y, neg_Y, args):
    X = F.normalize(X, dim=-1)
    X = X.unsqueeze(1) # B, 1, d

    Y = Y.unsqueeze(0)
    Y = torch.cat([Y, neg_Y], dim=0) # T, B, d
    Y = Y.transpose(0, 1)  # B, T, d
    Y = F.normalize(Y, dim=-1)

    logits = torch.bmm(X, Y.transpose(1, 2)).squeeze()  # B*T
    B = X.size()[0]
    labels = torch.zeros(B).long().to(logits.device)  # B*1

    criterion = nn.CrossEntropyLoss()

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    confidence = logits
    CL_conf = confidence.max(dim=1)[0]
    CL_conf = CL_conf.cpu().numpy()

    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B
    return CL_loss, CL_conf, CL_acc


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

    accum_acc_list = [0 for _ in args.T_list]

    for batch in tqdm(dataloader):
        
        text = batch[0]
        molecule_data = batch[1]
        neg_text = batch[2]
        neg_molecule_data = batch[3]

        text_repr = get_text_repr(text)
        molecule_repr = get_molecule_repr(molecule_data.to(device))

        if test_mode == "given_text":
            neg_molecule_repr = [get_molecule_repr(neg_molecule_data[idx].to(device)) for idx in range(T_max)]
            neg_molecule_repr = torch.stack(neg_molecule_repr)
        
            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(text_repr, molecule_repr, neg_molecule_repr[:T-1], args)
                accum_acc_list[T_idx] += acc
            
        elif test_mode == "given_molecule":
            neg_text_repr = [get_text_repr(neg_text[idx]) for idx in range(T_max)]
            neg_text_repr = torch.stack(neg_text_repr)
            for T_idx, T in enumerate(args.T_list):
                _, _, acc = do_CL_eval(molecule_repr, text_repr, neg_text_repr[:T-1], args)
                accum_acc_list[T_idx] += acc

        else:
            raise Exception
    
    accum_acc_list = np.array(accum_acc_list)
    accum_acc_list /= len(dataloader)
    return accum_acc_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--text_type", type=str, default="SciBERT", choices=["SciBERT", "BioBERT", "SentenceBERT"])

    parser.add_argument("--dataspace_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="description", choices=["ATC", "description", "pharmacodynamics"])
    parser.add_argument("--input_model_dir", type=str, default="./model_checkpoints")
    parser.add_argument("--input_model_config", type=str, 
                        default="AMOLE.pth")

    parser.add_argument("--test_mode", type=str, default="given_molecule", choices=["given_text", "given_molecule"])

    ########## for optimization ##########
    parser.add_argument("--T_list", type=int, nargs="+", default=[4, 10, 20])
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
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    ##### prepare text model #####
    if args.text_type == "SciBERT":
        pretrained_SciBERT_folder = os.path.join(args.dataspace_path, 'PubChemSTM' ,'pretrained_SciBERT')
        text_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder)
        # TODO: check https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L1501
        text_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased', cache_dir=pretrained_SciBERT_folder).to(device)
        text_dim = 768
    elif args.text_type == "SentenceBERT":
        pretrained_SentenceBERT_folder = os.path.join(args.dataspace_path, 'PubChemSTM' ,'pretrained_SentenceBERT')
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

    test_mode = args.test_mode
    T_max = max(args.T_list) - 1

    dataset_folder = os.path.join(args.dataspace_path, "Drugbank")
    dataloader_class = pyg_DataLoader

    test_acc_lists = list()

    for i in range(5):
        
        seed_everything(i)

        if args.dataset == "ATC":
            dataset_class = DrugBank_Datasets_Graph_ATC

            prompt_template = "This molecule is for {}."
            full_file_name = "SMILES_ATC_5_full.txt"
            processed_dir_prefix = "ATC_full_5"  

            dataset = dataset_class(dataset_folder, full_file_name, processed_dir_prefix, neg_sample_size=T_max, prompt_template=prompt_template)
            dataloader = dataloader_class(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        
        else :
            
            dataset_class = DrugBank_Datasets_Graph_retrieval

            if args.dataset == "description":
                processed_dir_prefix = "molecule_description_removed_PubChem"
                template = "SMILES_description_removed_from_PubChem_{}.txt"

            elif args.dataset == "pharmacodynamics":
                processed_dir_prefix = "molecule_pharmacodynamics_removed_PubChem"
                template = "SMILES_pharmacodynamics_removed_from_PubChem_{}.txt"
        
            full_dataset = dataset_class(dataset_folder, 'full', neg_sample_size=T_max, processed_dir_prefix=processed_dir_prefix, template=template)
            dataloader = dataloader_class(full_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        test_acc_list = eval_epoch(dataloader)
        
        print("arguments\t", args)
        print(test_acc_list)

        test_acc_lists.append(test_acc_list)
    
    test_acc_lists = np.vstack(test_acc_lists)
    print("5 run acc:", test_acc_lists.mean(axis = 0))
    print("5 run std:", test_acc_lists.std(axis = 0))

    # Write experimental results
    WRITE_PATH = "results_ret/"
    os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
    f = open("results_ret/{}.txt".format(args.dataset), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write("{}".format(args))
    f.write("\n")
    f.write("5 run acc: {}".format(test_acc_lists.mean(axis = 0)))
    f.write("\n")
    f.write("5 run std: {}".format(test_acc_lists.std(axis = 0)))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()