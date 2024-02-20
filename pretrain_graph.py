import torch
import sys
import os
import os.path as osp
import time
import pandas as pd

from torch_geometric.loader import DataLoader
from tqdm import tqdm
from copy import deepcopy

from dataset import load_dataset
from model import *
from utils import *
from evaluation import eval_graph

EVAL_KEYS = ['AUROC', 'AUPRC', 'RecK']

def pretrain(model, train_loader, eval_loader, args, logger):
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)

    supervised_type = 'Semi-supervised' if not args.full_supervised else 'Full-supervised'
    eval_times = int(args.epoch // args.eval_epoch)
    best_scores = {}
    for key in EVAL_KEYS:
        best_scores[key] = np.zeros(eval_times)
        best_scores[f'{key}_std'] = np.zeros(eval_times)
    eval_idx = 0
    patience, min_loss = 0, sys.maxsize
    epoch_iter = tqdm(range(args.epoch))
    for epoch in epoch_iter:
        model.train()
        avg_loss, epoch_time = 0, 0

        for step, batch in enumerate(train_loader):
            batch = batch.to(args.device)
            features, edge_index = batch.x, batch.edge_index
            edge_index, _ = dropout_edge(edge_index.clone(), p=args.edge_dropout)
            edge_adj = edge_index
            
            start_time = time.time()
            if args.strategy == 'graphcl':
                loss = model.get_loss(features, edge_index, batch=batch.batch)
            elif args.strategy == 'graphinfomax':
                loss = model.get_loss(features, edge_adj, batch=batch.batch)
            elif args.strategy == 'graphgd':
                _, pos_emb, neg_emb = model(features, edge_adj)

                loss = model.get_loss(pos_emb, neg_emb)
            elif args.strategy == 'graphmae':
                loss = model.get_loss(features, edge_adj)
            elif args.strategy == 'dgi_mae':
                loss = model.get_loss(features, edge_adj, beta=args.beta, batch=batch.batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            end_time = time.time()
            epoch_time += end_time - start_time
        
        avg_loss /= (step + 1)

        if avg_loss < min_loss:
            min_loss = avg_loss
            patience = 0
        else:
            patience += 1

        if patience >= args.patience:
            break
        
        scheduler.step()

        epoch_iter.set_description(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}, Duration: {epoch_time:.2f}s")

        if (epoch + 1) % args.eval_epoch == 0:
            model.eval()
            eval_scores = eval_graph(model, eval_loader, args, semi=(not args.full_supervised))
            for key in best_scores.keys():
                best_scores[key][eval_idx] = eval_scores[key]

            auc, prc, reck = eval_scores['AUROC'], eval_scores['AUPRC'], eval_scores['RecK']
            eval_idx += 1

            logger.info(f'Epoch {epoch}, {supervised_type}, AUROC: {auc:.1f}, AUPRC: {prc:.1f}, RecK: {reck:.1f}')
    
    while eval_idx < eval_times:
        for key in best_scores.keys():
            best_scores[key][eval_idx] = np.nan

        eval_idx += 1

    return best_scores


if __name__ == '__main__':

    # Configurations
    args = build_args('graph')
    if args.use_cfg:
        supervised = 'full' if args.full_supervised else 'semi'
        config_path = osp.join('config', f'{args.strategy}_{supervised}_config.yml')
        args = load_best_configs(args, config_path)

    # GPU initialization
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(args.device) if args.cuda else 'cpu')

    # Create logger
    logger, model_info = create_logger(args)

    # Data preprocessing
    data = load_dataset(args.data_dir, args.dataset)
    in_dim = data.x.shape[1]

    # Data preprocessing
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    # Model preparation
    seed = args.seed[0]
    set_random_seed(seed)
    assert args.strategy in ['graphcl', 'graphinfomax', 'graphgd', 'nonparametric', 'graphmae', 'dgi_mae']
    if args.strategy == 'graphcl':
        model = GraphCL(in_dim, args.hid_dim, args.num_layer, kernel=args.kernel, drop_ratio=args.dropout,
                    act=args.act, norm=args.norm, concat=args.concat,
                    tau=args.tau, pooler=args.pooler).to(args.device)
    elif args.strategy == 'graphinfomax':
        model = GraphInfoMax(in_dim, args.hid_dim, args.num_layer, kernel=args.kernel, drop_ratio=args.dropout,
                    act=args.act, norm=args.norm, concat=args.concat).to(args.device)
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
    
    test_scores = pretrain(model, train_loader, eval_loader, args, logger)
    select_array = np.array(test_scores[args.metric])
    select_idx = np.nanargmax(select_array)
    
    test_avg_scores = {'name': args.strategy}
    for key in EVAL_KEYS:
        test_avg_scores[f'{args.dataset.lower()}-{key}-mean'] = test_scores[key][select_idx]
        test_avg_scores[f'{args.dataset.lower()}-{key}-std'] = test_scores[f'{key}_std'][select_idx]
    test_avg_scores[f'{args.dataset.lower()}-best-epoch'] = (select_idx + 1)*args.eval_epoch
    test_avg_scores = pd.DataFrame(test_avg_scores, index=[0])

    if args.full_supervised:
        result_folder = 'loop_result_full'
    else:
        result_folder = 'loop_result_semi'
    result_folder = osp.join(args.result_dir, result_folder)
    os.makedirs(result_folder, exist_ok=True)

    # Print & save results
    supervised_type = 'Semi-supervised' if not args.full_supervised else 'Full-supervised'
    auc = test_avg_scores[f'{args.dataset.lower()}-AUROC-mean'][0]
    prc = test_avg_scores[f'{args.dataset.lower()}-AUPRC-mean'][0]
    reck = test_avg_scores[f'{args.dataset.lower()}-RecK-mean'][0]
    logger.info(f'{supervised_type}, Final Reuslts, AUROC: {auc:.1f}, AUPRC: {prc:.1f}, RecK: {reck:.1f}')

    result_path = osp.join(result_folder, f'{args.dataset}_{args.strategy}_result.xlsx')
    test_avg_scores.transpose().to_excel(result_path)