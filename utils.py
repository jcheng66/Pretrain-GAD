
import os
import yaml
import random
import logging
import sys
import argparse
import itertools

import torch
import torch.nn as nn
import numpy as np
import os.path as osp
import torch.nn.functional as F

from functools import partial

from torch_geometric.utils import sort_edge_index, subgraph, dropout_edge, mask_feature, shuffle_node
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


# ======================================================================
#   Reproducibility
# ======================================================================

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.determinstic = True



# ======================================================================
#   Configuration functions
# ======================================================================

def build_args(task=None):

    parser = argparse.ArgumentParser(description='Pretrain-GAD')
    # General settings
    parser.add_argument("--strategy", type=str, default="graphinfomax", help="Pretrain model strategy")
    parser.add_argument("--kernel", type=str, default="gcn", help="GNN model type")
    parser.add_argument("--dataset", type=str, default="weibo", help="Dataset for this model")
    
    parser.add_argument("--data_dir", type=str, default="./dataset/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="./model/", help="Folder to save model")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Folder to save logger")
    parser.add_argument("--result_dir", type=str, default="./results/", help="Folder to save results")
    parser.add_argument("--plot_dir", type=str, default="./plots/", help="Folder to save plots")

    # Model Configuration settings
    parser.add_argument("--seed", type=int, nargs="+", default=[12], help="Random seed")
    parser.add_argument("--hid_dim", type=int, default=32, help="Hidden layer dimension")
    parser.add_argument("--num_layer", type=int, default=2, help="Number of hidden layer in main model")
    parser.add_argument("--act", type=str, default='relu', help="Activation function type")
    parser.add_argument("--norm", type=str, default="", help="Normlaization layer type")
    parser.add_argument("--concat", action="store_true", default=False, help="Indicator of where using all embeddings")
    # GraphCL
    parser.add_argument("--tau", type=float, default=0.2, help="Temperature parameter for contrastive loss")
    # GraphMAE
    parser.add_argument("--mask_ratio", type=float, default=0.5, help="Masking ratio for GraphMAE")
    parser.add_argument("--replace_ratio", type=float, default=0, help="Replace ratio for GraphMAE")

    # Dataset settings
    parser.add_argument("--unify", action="store_true", default=False, help="SVD unify feature dimension")
    parser.add_argument("--unify_dim", type=int, default=100, help="SVD reduction dimension")
    parser.add_argument("--hetero_ratio", type=float, default=0, help="Ratio of hetero edges to be removed")
    parser.add_argument("--anomaly_ratio", type=float, default=0, help="Ratio of anomaly to be removed")
    parser.add_argument("--semi_samples", type=int, default=0, help="Samples of semi supervised learning")
    parser.add_argument("--shuffle_ratio", type=float, default=1.0, help="Feature shuffle ratio in DGI")

    # Training settings
    parser.add_argument("--epoch", type=int, default=1000, help="The max number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate of optimizer")
    parser.add_argument("--l2", type=float, default=0, help="Coefficient of L2 penalty")
    parser.add_argument("--decay_rate", type=float, default=1, help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100, help="Decay step of learning rate")
    parser.add_argument("--eval_epoch", type=int, default=20, help="Number of evaluation epoch")
    parser.add_argument("--sparse", action='store_true', default=False, help="Indicator of sparse computation")
    parser.add_argument("--contrast_batch", type=int, default=32, help="Batch size for contrastive learning")
    parser.add_argument("--patience", type=int, default=50, help="Early stop patience for pretraining")
    parser.add_argument("--non_train", action='store_true', default=False)
    parser.add_argument("--beta", type=float, default=1, help="Balance parameter between contrastive & generative ssl")

    # Evaluation settings
    parser.add_argument("--down_epoch", type=int, default=200, help="The max number of epochs for finetune")
    parser.add_argument("--down_act", type=str, default="relu", help="Activation fucntion for downsream model")
    parser.add_argument("--down_num_layer", type=int, default=2, help="Number of layers for downsream model")
    parser.add_argument("--full_supervised", default=False, action="store_true")
    parser.add_argument("--metric", type=str, default='AUPRC', help="Evaluation metric")
    parser.add_argument("--classifier", type=str, default='mlp', help="Evaluation classifier")

    # Hyperparameters
    parser.add_argument("--norm_type", type=str, default='sym', help="Type of normalization of adjacency matrix")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout rate for node in training")
    parser.add_argument("--edge_dropout", type=float, default=0, help="Dropout rate for edge in training")

    # Auxiliary
    parser.add_argument("--save_model", action='store_true', default=False, help="Indicator to save trained model")
    parser.add_argument("--load_model", action='store_true', default=False, help="Indicator to load trained model")
    parser.add_argument("--save_embed", action="store_true", default=False, help="Indicator to save best embedding")
    parser.add_argument("--log", action='store_true', default=False, help="Indicator to write logger file")
    parser.add_argument("--use_cfg", action="store_true", default=False, help="Indicator to use best configurations")

    # GPU settings
    parser.add_argument("--no_cuda", action='store_true', default=False, help="Indicator of GPU availability")
    parser.add_argument("--device", type=int, default=0, help='Which gpu to use if any')

    if task == 'graph':
        parser.add_argument("--pooler", type=str, default="sum", help="Pooling function for graph embedding")
        parser.add_argument("--batch_size", type=int, default=32, help="The batch size of training")
        parser.add_argument("--num_workers", type=int, default=8, help="Workers for dataloader")
        
        parser.set_defaults(epoch=300)
        parser.set_defaults(eval_epoch=10)

    # Display settings
    args = parser.parse_args()

    return args


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        print("-------------------- Best args not found, use default args --------------------")
    else:
        configs = configs[args.dataset]

        for k, v in configs.items():
            if "lr" in k or "beta" in k:
                v = float(v)
            setattr(args, k, v)
        print("-------------------- Use best configs --------------------")

    return args

# ======================================================================
#   Logger functions
# ======================================================================


def create_logger(args, search=False):

    # Logger directory
    # os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(osp.join(args.log_dir, args.dataset), exist_ok=True)

    if not search:
        model_info = '{}_{}_{}_{}_{}'.format(args.strategy, args.kernel, args.num_layer, args.act, args.hid_dim)

        if args.norm:
            model_info += '_{}'.format(args.norm)
    else:
        model_info = '{}_param_search'.format(args.strategy)

    log_file = f'{model_info}.txt'

    log_path = osp.join(args.log_dir, args.dataset, log_file)
    log_format = '%(levelname)s %(asctime)s - %(message)s'
    log_time_format = '%Y-%m-%d %H:%M:%S'
    
    if args.log:
        log_handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        log_handlers = [
            logging.StreamHandler(sys.stdout)
        ]
    logging.basicConfig(
        format=log_format,
        datefmt=log_time_format,
        level=logging.INFO,
        handlers=log_handlers
    )
    logger = logging.getLogger()

    return logger, model_info


def create_logger_super(args):

    os.makedirs(osp.join(args.log_dir, args.dataset), exist_ok=True)

    model_info = '{}_{}_{}'.format(args.kernel, args.num_layer, args.hid_dim)

    log_path = osp.join(args.log_dir, args.dataset, f'supervised_{args.kernel}.txt')
    log_format = '%(levelname)s %(asctime)s - %(message)s'
    log_time_format = '%Y-%m-%d %H:%M:%S'
    
    if args.log:
        log_handlers = [
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    else:
        log_handlers = [
            logging.StreamHandler(sys.stdout)
        ]
    logging.basicConfig(
        format=log_format,
        datefmt=log_time_format,
        level=logging.INFO,
        handlers=log_handlers
    )
    logger = logging.getLogger()

    return logger, model_info



# ======================================================================
#   Model activation/normalization creation function
# ======================================================================

def obtain_act(name=None):
    """
    Return activation function module
    """
    if name == 'relu':
        act = nn.ReLU(inplace=True)
    elif name == "gelu":
        act = nn.GELU()
    elif name == "prelu":
        act = nn.PReLU()
    elif name == "elu":
        act = nn.ELU()
    elif name == "leakyrelu":
        act = nn.LeakyReLU()
    elif name == "tanh":
        act = nn.Tanh()
    elif name == "sigmoid":
        act = nn.Sigmoid()
    elif name is None:
        act = nn.Identity()
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return act


def obtain_norm(name):
    """
    Return normalization function module
    """
    if name == "layernorm":
        norm = nn.LayerNorm
    elif name == "batchnorm":
        norm = nn.BatchNorm1d
    elif name == "instancenorm":
        norm = partial(nn.InstanceNorm1d, affine=True, track_running_stats=True)
    else:
        raise NotImplementedError("{} is not implemented.".format(name))

    return norm


def obtain_pooler(name):
    """
    Return pooling function module
    """
    if name == 'mean':
        pooler = global_mean_pool
    elif name == 'sum':
        pooler = global_add_pool
    elif name == 'max':
        pooler = global_max_pool

    return pooler


# ======================================================================
#   Data augmentation funciton
# ======================================================================

def graphcl_augmentation(features, edge_index, batch=None, n=None):

    if n is None:
        n = np.random.randint(2)
    if n == 0:
        edge_index, _ = dropout_edge(edge_index.clone(), p=0.1, force_undirected=True)
    elif n == 1:
        features, _ = mask_feature(features.clone(), p=0.2, mode='col')
    else:
        print('sample error')
        assert False
        
    return features, edge_index


def infomax_corruption(features, batch=None, ratio=1.0):
    
    if batch is None:
        if ratio == 1:
            shuffle_features, _ = shuffle_node(features, batch)
        else:
            shuffle_features = features.clone()
            if ratio > 0:
                num_shuffle = int(features.size(0) * ratio)
                perm_1 = torch.randperm(features.size(0), device=features.device)
                shuffle_idx = perm_1[:num_shuffle]
                perm_2 = torch.randperm(shuffle_idx.size(0), device=shuffle_idx.device)
                to_shuffle = shuffle_idx[perm_2]
                shuffle_features[shuffle_idx, :] = shuffle_features[to_shuffle, :]
    else:
        shuffle_features, _ = shuffle_node(features, batch)

    return shuffle_features


# ======================================================================
#   Data augmentation funciton
# ======================================================================

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


# ======================================================================
#   Parameters search settings
# ======================================================================

def get_params_range(task='node'):

    params = []
    space = NODE_PARAM_SPACE if task == 'node' else GRAPH_PARAM_SPACE
    for key, value in space.items():
        params.append(value)
    
    params_range = [param for param in itertools.product(*params)]
    return params_range

def set_params(args, params, task='node'):

    key_config = {}
    space = NODE_PARAM_SPACE if task == 'node' else GRAPH_PARAM_SPACE
    for key, value in zip(space.keys(), params):
        setattr(args, key, value)
        key_config[key] = value
    
    return key_config


NODE_PARAM_SPACE = {
    "kernel": ['gcn', 'gin'],
    "num_layer": [1, 2],
    "act": ["tanh", "relu", "leakyrelu"],
    "norm": ["", "batchnorm"],
}

GRAPH_PARAM_SPACE = {
    "kernel": ['gcn', 'gin'],
    "num_layer": [1, 2],
    "act": ["tanh", "relu", "leakyrelu"],
    "norm": ["", "batchnorm", "layernorm"],
}


# ======================================================================
#   Dataset adjustment functions
# ======================================================================

def remove_hetero_edges(dataset, ratio=0):

    if ratio > 0:
        homo_idx = torch.where(dataset.y[dataset.edge_index[0]] == dataset.y[dataset.edge_index[1]])[0]
        hetero_idx = torch.where(dataset.y[dataset.edge_index[0]] != dataset.y[dataset.edge_index[1]])[0]
        
        num_hetero = hetero_idx.shape[0]
        set_random_seed(0)
        select_idx = torch.randperm(num_hetero)[:int(num_hetero*(1 - ratio))]

        concat_idx = torch.concat([homo_idx, hetero_idx[select_idx]])
        edit_edge_index = dataset.edge_index[:, concat_idx]
        edit_edge_index = sort_edge_index(edit_edge_index)

        dataset.edge_index = edit_edge_index
    
    return dataset


def remove_anomaly(dataset, ratio=0):

    if ratio > 0:
        anomaly_idx = torch.where(dataset.y == 1)[0]
        num_anomaly = anomaly_idx.shape[0]
        set_random_seed(0)
        anomaly_idx = anomaly_idx[torch.randperm(num_anomaly)]
        rand_idx = anomaly_idx[:int((1 - ratio)*num_anomaly)]
        remove_idx = anomaly_idx[int((1 - ratio)*num_anomaly):]
        select_idx = torch.concat([torch.where(dataset.y == 0)[0], rand_idx], dim=0)
        select_idx = select_idx.sort()[0]
        
        edge_index = subgraph(select_idx, dataset.edge_index)[0]
        dataset.edge_index = edge_index

        dataset.select_idx = select_idx
        dataset.sub_edge_index = subgraph(select_idx, dataset.edge_index, relabel_nodes=True)[0]

        dataset.train_masks[remove_idx, :] = False
        dataset.val_masks[remove_idx, :] = False
        dataset.test_masks[remove_idx, :] = False
        dataset.semi_train_masks[remove_idx, :] = False
        dataset.semi_val_masks[remove_idx, :] = False
        dataset.semi_test_masks[remove_idx, :] = False

    return dataset