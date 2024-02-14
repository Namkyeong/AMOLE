import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--data_path", type=str, default="./data/PubChemSTM")
    parser.add_argument("--checkpoint_path", type=str, default="./model_checkpoints")

    parser.add_argument("--dataset", type=str, default="TanimotoSTM", choices=["TanimotoSTM"])
    parser.add_argument("--model", type=str, default="AMOLE")

    parser.add_argument("--batch_size", type=int, default=45)
    parser.add_argument("--text_lr", type=float, default=1e-5)
    parser.add_argument("--mol_lr", type=float, default=1e-5)
    parser.add_argument("--text_lr_scale", type=float, default=1.0)
    parser.add_argument("--mol_lr_scale", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--decay", type=float, default=0)

    parser.add_argument("--representation_frozen", dest='representation_frozen', action='store_true')
    parser.add_argument('--no_representation_frozen', dest='representation_frozen', action='store_false')
    parser.set_defaults(representation_frozen=False)

    parser.add_argument('--normalize', dest='normalize', action='store_true')
    parser.add_argument('--no_normalize', dest='normalize', action='store_false')
    parser.set_defaults(normalize=True)

    # For SciBERT
    parser.add_argument("--lm", type=str, default="SciBERT", choices=["SciBERT"])
    parser.add_argument("--max_seq_len", type=int, default=512)

    # For Graph Neural Networks
    parser.add_argument("--pretrain_gnn_mode", type=str, default="GraphMVP_G", choices=["GraphMVP_G"])
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument('--graph_pooling', type=str, default='mean')

    # For Contrastive Learning
    parser.add_argument("--SSL_emb_dim", type=int, default=256)
    parser.add_argument("--CL_neg_samples", type=int, default=1)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--target_T", type=float, default=0.1)

    # For Augmentation
    parser.add_argument("--p_aug", type=float, default=0.5)
    parser.add_argument("--num_cand", type=int, default=50)

    # For Expertise Transfer
    parser.add_argument("--alpha", type=float, default=1.0)
    
    return parser.parse_known_args()


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device', 'data_path', 'checkpoint_path', 'writer', 
                        'batch_size', 'text_lr_scale', 'mol_lr_scale', 'num_workers', 'decay', 'max_seq_len', 
                        'pretrain_gnn_mode', 'gnn_emb_dim', 'num_layer', 'JK', 'gnn_type', 'graph_pooling',
                        'SSL_emb_dim', 'CL_neg_samples', 'normalize', 
                        'eval_task', 'test_mode', 'T_list']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)