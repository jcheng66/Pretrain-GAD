import torch
import torch.nn.functional as F
import numpy as np

from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, average_precision_score

from model import LinearPred
from utils import *


# ======================================================================
#   MLP Evaluator
# ======================================================================

def eval_MLP(embeds, labels, train_mask, val_mask, test_mask,
                  weight, seed, args, return_preds=False):

    K = labels[test_mask].sum()
    set_random_seed(seed)
    model_pred = LinearPred(embeds.shape[1], args.hid_dim, 2, args.down_num_layer, act=args.down_act).to(args.device)
    optimizer_pred = torch.optim.Adam(model_pred.parameters(), lr=0.01)

    val_best = 0
    val_score = {}
    test_probs = None
    patience = 0
    for j in range(args.down_epoch):
        model_pred.train()
        optimizer_pred.zero_grad()

        logits = model_pred(embeds[train_mask])
        loss = F.cross_entropy(logits, labels[train_mask], weight=weight)

        loss.backward()
        optimizer_pred.step()

        model_pred.eval()
        with torch.no_grad():
            val_probs = model_pred(embeds[val_mask]).softmax(dim=1)[:, 1]
        # val_score['AUROC'] = roc_auc_score(labels[val_mask].cpu(), val_probs.cpu())
        val_score['AUPRC'] = average_precision_score(labels[val_mask].cpu(), val_probs.cpu())

        if val_score[args.metric] >= val_best:
            val_best = val_score[args.metric]
            with torch.no_grad():
                test_probs = model_pred(embeds[test_mask]).softmax(dim=1)[:, 1]
                if return_preds:
                    best_probs = model_pred(embeds).softmax(dim=1)[:, 1]
            ep_b = j
            patience = 0
        else:
            patience += 1
        
        if patience >= 30:
            break
    
    auc_test = roc_auc_score(labels[test_mask].cpu(), test_probs.cpu()) * 100
    prc_test = average_precision_score(labels[test_mask].cpu(), test_probs.cpu()) * 100
    reck_test = (sum(labels[test_mask][test_probs.argsort()[-K:]]) / K).cpu().item() * 100

    if return_preds:
        return auc_test, prc_test, reck_test, ep_b, best_probs
    else:
        return auc_test, prc_test, reck_test, ep_b


# ======================================================================
#   Node-level GAD evaluation
# ======================================================================

def eval_node(model, data, k, args, semi=True, node2vec=False):

    features = data.x.to(args.device)
    edge_index = data.edge_index
    edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.shape[1]), 
                            sparse_sizes=(data.x.size(0), data.x.size(0))).to(args.device)
    labels = data.y.long().to(args.device)

    if semi:
        train_mask = data.semi_train_masks[:, k]
        val_mask = data.semi_val_masks[:, k]
        test_mask = data.semi_test_masks[:, k]
    else:
        train_mask = data.train_masks[:, k]
        val_mask = data.val_masks[:, k]
        test_mask = data.test_masks[:, k]

    weight = torch.tensor(
        [1, (1 - labels[train_mask]).sum().item()/labels[train_mask].sum().item()],
        device=labels.device
    )

    with torch.no_grad():
        if isinstance(model, list):
            embeds = []
            for m_model in model:
                if not node2vec:
                    embed = m_model.embed(features, edge_index)
                else:
                    embed = m_model().clone().detach()
                embeds.append(embed)
            
            embeds = torch.concat(embeds, dim=1)
        else:
            if not node2vec:
                embeds = model.embed(features, edge_index)
            else:
                embeds = model().clone().detach()

    # Evaluation
    test_results = eval_MLP(embeds, labels, train_mask, val_mask, test_mask, weight, k, args, True)
    auc, prc, reck, best_epoch, pred = test_results
        
    results = {'AUROC': auc, 'AUPRC': prc, 'RecK': reck}

    return results


# ======================================================================
#   Graph-level GAD evaluation
# ======================================================================

def eval_graph(model, dataloader, args, semi=True, node2vec=False, return_preds=False):

    pooler = obtain_pooler(args.pooler)
    # Obtain graph embeddings
    embed_list = []
    y_list = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            batch = batch.to(args.device)
            if not node2vec:
                embed = model.embed(batch.x, batch.edge_index)
            else:
                embed = model().clone().detach()
            embed = pooler(embed, batch.batch)

            embed_list.append(embed)
            y_list.append(batch.y)
    
    embed = torch.concat(embed_list, dim=0)
    labels = torch.concat(y_list, dim=0)

    # Downstream evaluation
    data = dataloader.dataset
    aucs, prcs, recks = [], [], []
    best_epochs = []
    preds = []
    for k in range(5):

        if semi:
            train_mask = data.semi_train_masks[:, k]
            val_mask = data.semi_val_masks[:, k]
            test_mask = data.semi_test_masks[:, k]
        else:
            train_mask = data.train_masks[:, k]
            val_mask = data.val_masks[:, k]
            test_mask = data.test_masks[:, k]

        weight = torch.tensor(
            [1, (1 - labels[train_mask]).sum().item()/labels[train_mask].sum().item()],
            device=labels.device
        )
        
        test_results = eval_MLP(embed, labels, train_mask, val_mask, test_mask, weight, k, args, return_preds)
        auc, prc, reck, best_epoch = test_results[:4]
        aucs.append(auc)
        prcs.append(prc)
        recks.append(reck)
        best_epochs.append(best_epoch)
        if return_preds:
            preds.append(test_results[-1])
    
    results = {
        'AUROC': np.mean(aucs), 'AUPRC': np.mean(prcs), 'RecK': np.mean(recks),
        'AUROC_std': np.std(aucs), 'AUPRC_std': np.std(prcs), 'RecK_std': np.std(recks)
    }

    if return_preds:
        preds = torch.stack(preds).T
        return results, preds
    else:
        return results

