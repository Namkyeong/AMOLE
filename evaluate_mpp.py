import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torch_DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader

from tensorboardX import SummaryWriter

from datasets import MoleculeNetGraphDataset
from utils.splitters import scaffold_split
from layers import GNN, GNN_graphpred


def get_num_task_and_type(dataset):
    if dataset in ["esol", "freesolv", "lipophilicity"]:
        return 1, "regression"
    elif dataset in ["hiv", "bace", "bbbp"]:
        return 1, "classification"
    elif dataset == "tox21":
        return 12, "classification"
    elif dataset == "pcba":
        return 92, "classification"
    elif dataset == "muv":
        return 17, "classification"
    elif dataset == "toxcast":
        return 617, "classification"
    elif dataset == "sider":
        return 27, "classification"
    elif dataset == "clintox":
        return 2, "classification"
    raise ValueError("Invalid dataset name.")


def train_classification(model, device, loader, optimizer):
    if args.training_mode == "fine_tuning":
        model.train()
    else:
        model.eval()
    linear_model.train()
    total_loss = 0

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        
        batch = batch.to(device)
        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()
        y = batch.y.view(pred.shape).to(device).float()

        is_valid = y ** 2 > 0
        loss_mat = criterion(pred, (y + 1) / 2)
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_classification(model, device, loader):
    model.eval()
    linear_model.eval()
    y_true, y_scores = [], []

    if args.verbose:
        L = tqdm(loader)
    else:
        L = loader
    for step, batch in enumerate(L):
        batch = batch.to(device)
        molecule_repr, _ = model(batch)
        pred = linear_model(molecule_repr)
        pred = pred.float()
        y = batch.y.view(pred.shape).to(device).float()

        y_true.append(y)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print("{} is invalid".format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--training_mode", type=str, default="fine_tuning", choices=["fine_tuning", "linear_probing"])
    parser.add_argument("--molecule_type", type=str, default="Graph", choices=["SMILES", "Graph"])

    ########## for dataset and split ##########
    parser.add_argument("--dataspace_path", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--split", type=str, default="scaffold")
    parser.add_argument('--split_ratio', nargs='?', default='[0.8, 0.1, 0.1]')

    ########## for optimization ##########
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scale", type=float, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--schedule", type=str, default="cycle")
    parser.add_argument("--warm_up_steps", type=int, default=10)

    ########## for 2D GNN ##########
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    ########## for saver ##########
    parser.add_argument("--eval_train", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--input_model_dir", type=str, default="./model_checkpoints")
    parser.add_argument("--input_model_config", type=str, 
                        default="AMOLE.pth")
    parser.add_argument("--output_model_dir", type=str, default=None)

    args = parser.parse_args()
    print("arguments\t", args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")

    num_tasks, task_mode = get_num_task_and_type(args.dataset)
    dataset_folder = os.path.join(args.dataspace_path, "MoleculeNet_data", args.dataset)

    dataset = MoleculeNetGraphDataset(dataset_folder, args.dataset)
    dataloader_class = pyg_DataLoader
    use_pyg_dataset = True

    assert args.split == "scaffold"
    print("split via scaffold")
    smiles_list = pd.read_csv(
        dataset_folder + "/processed/smiles.csv", header=None)[0].tolist()

    split_ratio = eval(args.split_ratio)
    
    train_dataset, valid_dataset, test_dataset = scaffold_split(
        dataset, smiles_list, null_value=0, frac_train=split_ratio[0],
        frac_valid=split_ratio[1], frac_test=split_ratio[2], pyg_dataset=use_pyg_dataset)
    
    train_loader = dataloader_class(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = dataloader_class(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = dataloader_class(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    molecule_node_model = GNN(
        num_layer=args.num_layer, emb_dim=args.gnn_emb_dim,
        JK=args.JK, drop_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type)
    model = GNN_graphpred(
        num_layer=args.num_layer, emb_dim=args.gnn_emb_dim, JK=args.JK, graph_pooling=args.graph_pooling,
        num_tasks=1, molecule_node_model=molecule_node_model) 
    molecule_dim = args.gnn_emb_dim
    if args.input_model_config is not None:
        print("Start from pretrained model (MoleculeSTM) in {}.".format(args.input_model_config))
        input_model_path = os.path.join(args.input_model_dir, "molecule", args.input_model_config)
        state_dict = torch.load(input_model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        print("Start from randomly initialized GNN.")

    _train_roc_list, _val_roc_list, _test_roc_list = [], [], []

    for seed in range(5):

        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = model.to(device)
        linear_model = nn.Linear(molecule_dim, num_tasks).to(device)

        # set up optimizer
        if args.training_mode == "fine_tuning":
            model_param_group = [
                {"params": model.parameters()},
                {"params": linear_model.parameters(), 'lr': args.lr * args.lr_scale}
            ]
        else:
            model_param_group = [
                {"params": linear_model.parameters(), 'lr': args.lr * args.lr_scale}
            ]
        optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.weight_decay)

        if task_mode == "classification":
            train_func = train_classification
            eval_func = eval_classification

            train_roc_list, val_roc_list, test_roc_list = [], [], []
            train_acc_list, val_acc_list, test_acc_list = [], [], []
            best_val_roc, best_val_idx = -1, 0
            criterion = nn.BCEWithLogitsLoss(reduction="none")

            for epoch in range(1, args.epochs + 1):
                loss_acc = train_func(model, device, train_loader, optimizer)
                print("Epoch: {}\nLoss: {}".format(epoch, loss_acc))

                if args.eval_train:
                    train_roc, train_acc, train_target, train_pred = eval_func(model, device, train_loader)
                else:
                    train_roc = train_acc = 0
                val_roc, val_acc, val_target, val_pred = eval_func(model, device, val_loader)
                test_roc, test_acc, test_target, test_pred = eval_func(model, device, test_loader)

                train_roc_list.append(train_roc)
                train_acc_list.append(train_acc)
                val_roc_list.append(val_roc)
                val_acc_list.append(val_acc)
                test_roc_list.append(test_roc)
                test_acc_list.append(test_acc)
                print("train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(train_roc, val_roc, test_roc))
                print()

                if val_roc > best_val_roc:
                    best_val_roc = val_roc
                    best_val_idx = epoch - 1

            print("[Seed: {}] Final train: {:.6f}\tval: {:.6f}\ttest: {:.6f}".format(seed, train_roc_list[-1], val_roc_list[-1], test_roc_list[-1]))

        _train_roc_list.append(train_roc_list[-1])
        _val_roc_list.append(val_roc_list[-1])
        _test_roc_list.append(test_roc_list[-1])

    # Write experimental results
    WRITE_PATH = "results_prop/"
    os.makedirs(WRITE_PATH, exist_ok=True) # Create directory if it does not exist
    f = open("results_prop/{}_{}.txt".format(args.training_mode, args.dataset), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write("lr: {} / ratio: {} / model: {}".format(args.lr, args.split_ratio, args.input_model_config))
    f.write("\n")
    f.write("[Valid] 5 run acc: {:.4f} ({:.4f})".format(np.mean(_val_roc_list), np.std(_val_roc_list)))
    f.write("\n")
    f.write("[Test] 5 run acc: {:.4f} ({:.4f})".format(np.mean(_test_roc_list), np.std(_test_roc_list)))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()