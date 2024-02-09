import torch
import sys
import os
import os.path as osp
import pandas as pd

from torch_geometric.utils import to_torch_sparse_tensor
from torch_geometric.transforms import SVDFeatureReduction
from tqdm import tqdm
from copy import deepcopy

from dataset import load_dataset
from model import *
from utils import *
from evaluation import eval_node_loop

EVAL_KEYS = ['AUROC', 'AUPRC', 'RecK']
KEYS = ['1_hop', '2_hop', 'only_2_hop', 'remain']

def pretrain(model, data, k, args, logger):

    # Initialization
    edge_index = data.edge_index.to(args.device)
    features = data.x.to(args.device)
    edge_adj = to_torch_sparse_tensor(edge_index, size=features.size(0))
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    supervised_type = 'Semi-supervised' if not args.full_supervised else 'Full-supervised'
    eval_times = int(args.epoch // args.eval_epoch)
    best_scores = {key: np.zeros(eval_times) for key in EVAL_KEYS}
    reach_scores = {key: np.zeros(eval_times) for key in KEYS}
    eval_idx = 0
    patience, min_loss = 0, sys.maxsize
    epoch_iter = tqdm(range(args.epoch))
    for epoch in epoch_iter:
        model.train()
        optimizer.zero_grad()

        if args.strategy == 'graphcl':
            loss = model.get_loss(features, edge_adj)
        elif args.strategy == 'graphinfomax':
            loss = model.get_loss(features, edge_adj)
        elif args.strategy == 'graphgd':
            _, pos_emb, neg_emb = model(features, edge_adj)

            loss = model.get_loss(pos_emb, neg_emb)
        elif args.strategy == 'graphmae':
            loss = model.get_loss(features, edge_adj)
        elif args.strategy == 'dgi_mae':
            loss = model.get_loss(features, edge_adj, beta=args.beta)

        if loss < min_loss:
            min_loss = loss.item()
            patience = 0
        else:
            patience += 1

        if patience >= args.patience:
            break

        loss.backward()
        optimizer.step()
        scheduler.step()

        epoch_iter.set_description(f"Epoch {epoch}, Train Loss: {loss.item():.4f}")

        if (epoch + 1) % args.eval_epoch == 0:
            model.eval()
            # if args.classifier == 'mlp':
            eval_scores, eval_reach_scores = eval_node_loop(model, data, k, args, semi=(not args.full_supervised))
            # else:
            #     eval_scores, preds, reach_score  = eval_classifier(model, data, args, semi=(not args.full_supervised), classifier=args.classifier)
            for key in best_scores.keys():
                best_scores[key][eval_idx] = eval_scores[key]
            
            for key in reach_scores.keys():
                reach_scores[key][eval_idx] = eval_reach_scores[key]

            auc, prc, reck = eval_scores['AUROC'], eval_scores['AUPRC'], eval_scores['RecK']
            eval_idx += 1
            
            logger.info(f'Split {k}, Epoch {epoch}, {supervised_type}, AUROC: {auc:.1f}, AUPRC: {prc:.1f}, RecK: {reck:.1f}')

    while eval_idx < eval_times:
        for key in best_scores.keys():
            best_scores[key][eval_idx] = np.nan

        for key in reach_scores.keys():
            reach_scores[key][eval_idx] = np.nan

        eval_idx += 1

    return best_scores, reach_scores


if __name__ == '__main__':

    # Configurations
    args = build_args()
    if args.use_cfg:
        supervised = 'full' if args.full_supervised else 'semi'
        config_path = osp.join('config', f'{args.strategy}_{supervised}_config.yml')
        args = load_best_configs(args, config_path)

    # GPU initialization
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')

    # Data preprocessing
    data = load_dataset(args.data_dir, args.dataset, args.semi_samples, args.full_supervised)
    data = remove_hetero_edges(data, args.hetero_ratio)
    data = remove_anomaly(data, args.anomaly_ratio)
    if args.unify:
        reducer = SVDFeatureReduction(out_channels=args.unify_dim)
        data = reducer(data)
    in_dim = data.x.shape[1]

    # Create logger
    logger, model_info = create_logger(args)

    test_scores = {key: [] for key in EVAL_KEYS}
    reach_scores = {key: [] for key in KEYS}

    # Model preparation
    for k in range(10):
        set_random_seed(k)
        assert args.strategy in ['graphcl', 'graphinfomax', 'graphgd', 'nonparametric', 'graphmae', 'dgi_mae']
        if args.strategy == 'graphcl':
            model = GraphCL(in_dim, args.hid_dim, args.num_layer, kernel=args.kernel, drop_ratio=args.dropout,
                        act=args.act, norm=args.norm, concat=args.concat, tau=args.tau).to(args.device)
        elif args.strategy == 'graphinfomax':
            model = GraphInfoMax(in_dim, args.hid_dim, args.num_layer, kernel=args.kernel, drop_ratio=args.dropout,
                        act=args.act, norm=args.norm, concat=args.concat, shuffle_ratio=args.shuffle_ratio).to(args.device)
        elif args.strategy == 'graphgd':
            model = GraphGD(in_dim, args.hid_dim, args.num_layer, kernel=args.kernel, drop_ratio=args.dropout,
                        act=args.act, norm=args.norm, concat=args.concat).to(args.device)
        elif args.strategy == 'graphmae':
            model = GraphMAE(in_dim, args.hid_dim, args.num_layer, kernel=args.kernel, drop_ratio=args.dropout,
                        act=args.act, norm=args.norm, mask_ratio=args.mask_ratio, 
                        replace_ratio=args.replace_ratio, concat=args.concat).to(args.device)
        elif args.strategy == 'nonparametric':
            model = NonParametric(in_dim, args.hid_dim, args.num_layer, act=args.act, concat=args.concat).to(args.device)
        elif args.strategy == 'dgi_mae':
            model = DGI_MAE(in_dim, args.hid_dim, args.num_layer, kernel=args.kernel, drop_ratio=args.dropout,
                        act=args.act, norm=args.norm, mask_ratio=args.mask_ratio, 
                        replace_ratio=args.replace_ratio, concat=args.concat).to(args.device)
        

        test_score, reach_score = pretrain(model, data, k, args, logger)
        
        for key in test_scores.keys():
            test_scores[key].append(test_score[key])
        
        if not args.full_supervised:
            for key in reach_scores.keys():
                reach_scores[key].append(reach_score[key])
    
    select_array = np.array(test_scores[args.metric])
    select_array = np.nanmean(select_array, axis=0)
    select_idx = np.nanargmax(select_array)
    
    test_avg_scores = {'name': args.strategy}
    for key in test_scores.keys():
        metrics = np.array(test_scores[key])
        metrics = metrics[:, select_idx]
        test_avg_scores[f'{args.dataset.lower()}-{key}-mean'] = np.nanmean(metrics)
        test_avg_scores[f'{args.dataset.lower()}-{key}-std'] = np.nanstd(metrics)
    test_avg_scores[f'{args.dataset.lower()}-best-epoch'] = (select_idx + 1)*args.eval_epoch
    test_avg_scores = pd.DataFrame(test_avg_scores, index=[0])

    if not args.full_supervised:
        reach_avg_scores = {'name': args.strategy}
        for key in reach_scores.keys():
            metrics = np.array(reach_scores[key])
            metrics = metrics[:, select_idx]
            reach_avg_scores[f'{args.dataset.lower()}-{key}'] = np.nanmean(metrics)
        reach_avg_scores[f'{args.dataset.lower()}-best-epoch'] = (select_idx + 1)*args.eval_epoch
        reach_avg_scores = pd.DataFrame(reach_avg_scores, index=[0])

    if args.full_supervised:
        result_folder = 'loop_result_full'
    else:
        result_folder = 'loop_result' if args.semi_samples == 0 else f'loop_result_{args.semi_samples}'
        if args.shuffle_ratio != 1:
            result_folder += f'_{args.shuffle_ratio}'
        result_folder = osp.join(args.result_dir, result_folder)
    os.makedirs(result_folder, exist_ok=True)
    
    result_path = osp.join(result_folder, f'{args.dataset}_{args.strategy}_result.xlsx')
    test_avg_scores.transpose().to_excel(result_path)
    if not args.full_supervised:
        reach_path = osp.join(result_folder, f'{args.dataset}_{args.strategy}_reach.xlsx')
        reach_avg_scores.transpose().to_excel(reach_path)