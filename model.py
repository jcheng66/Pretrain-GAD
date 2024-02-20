import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv, SAGEConv, TransformerConv, GATConv, SimpleConv, global_add_pool, global_mean_pool
from torch_geometric.nn.inits import uniform
from torch_geometric.utils import is_torch_sparse_tensor, to_edge_index, to_torch_sparse_tensor

from utils import obtain_act, obtain_norm, obtain_pooler, infomax_corruption, sce_loss, graphcl_augmentation


# ======================================================================
#   Basics
# ======================================================================

class LinearPred(nn.Module):

    def __init__(self, in_dim, emb_dim, out_dim, num_layer, act=None):
        super().__init__()

        self.linears = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.num_layer = num_layer

        for i in range(self.num_layer):
            if i == 0:
                if self.num_layer == 1:
                    self.linears.append(nn.Linear(in_dim, out_dim))
                else:
                    self.linears.append(nn.Linear(in_dim, emb_dim))
            elif i == self.num_layer - 1:
                self.linears.append(nn.Linear(emb_dim, out_dim))
            else:
                self.linears.append(nn.Linear(emb_dim, emb_dim))
            if i != self.num_layer - 1:
                self.acts.append(obtain_act(act))
            else:
                self.acts.append(obtain_act(None))

            # Initialize parameters
            nn.init.xavier_uniform_(self.linears[-1].weight)
            if self.linears[-1].bias is not None:
                self.linears[-1].bias.data.fill_(0.0)

    def forward(self, x):
        
        for i in range(self.num_layer):
            x = self.acts[i](self.linears[i](x))
        
        return x

    def get_embedding(self, x):
        
        if self.num_layer > 1:
            for i in range(self.num_layer - 1):
                x = self.acts[i](self.linears[i](x))
            
            return x
        else:
            return x


class Encoder(nn.Module):

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm='batchnorm', concat=False, last_act=True,
                 aggr='mean', pooler='sum'):
        super().__init__()

        self.num_layer = num_layer
        self.emb_dim = [in_dim] + [emb_dim] * num_layer

        self.drop_ratio = drop_ratio
        self.norm = norm
        self.concat = concat

        self.pooler = obtain_pooler(pooler)

        self.encs = torch.nn.ModuleList()
        self.acts = torch.nn.ModuleList()
        if self.norm:
            self.norms = torch.nn.ModuleList()

        for i in range(self.num_layer):
            if kernel == 'gcn':
                conv = GCNConv(self.emb_dim[i], self.emb_dim[i + 1], normalize=True, add_self_loops=True, cached=False)
            elif kernel == 'gin':
                conv = GINConv(LinearPred(self.emb_dim[i], self.emb_dim[i + 1], self.emb_dim[i + 1], 1), aggr=aggr)
            elif kernel == 'gin2':
                conv = GINConv(LinearPred(self.emb_dim[i], self.emb_dim[i + 1], self.emb_dim[i + 1], 2), aggr=aggr)
            elif kernel == 'sage':
                conv = SAGEConv(self.emb_dim[i], self.emb_dim[i + 1])
            elif kernel == 'transformer':
                conv = TransformerConv(self.emb_dim[i], self.emb_dim[i + 1])
            elif kernel == 'gat':
                conv = GATConv(self.emb_dim[i], self.emb_dim[i + 1])
            elif kernel == 'simpleconv':
                conv = SimpleConv(aggr='mean', combine_root='self_loop')
            self.encs.append(conv)
            if i == self.num_layer - 1 and not last_act:
                act = None
            self.acts.append(obtain_act(act))

            if self.norm:
                self.norms.append(obtain_norm(self.norm)(self.emb_dim[i + 1]))
        
    def forward(self, x, edge_index, edge_weight=None, batch=None):

        xs = []
        for i in range(self.num_layer):
            x = self.encs[i](x, edge_index, edge_weight)
            x = self.norms[i](x) if self.norm else x
            x = F.dropout(self.acts[i](x), self.drop_ratio, training=self.training)
            xs.append(x)
        
        if batch is not None:
            xs = [self.pooler(x, batch) for x in xs]
        
        if self.concat:
            x = torch.concat(xs, dim=1)
        else:
            x = xs[-1]

        return x
    
    def embed(self, x, edge_index, edge_weight=None, batch=None):
        
        raw = x if batch is None else global_add_pool(x, batch)
        embeds = self.forward(x, edge_index, edge_weight, batch)
        if self.concat:
            embeds = torch.concat([raw, embeds], dim=1)
        
        return embeds


# ======================================================================
#   Contrastive SSL
# ======================================================================


class GraphInfoMax(nn.Module):

    EPS = 1e-15

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None, concat=False, shuffle_ratio=1.0):
        super().__init__()

        self.emd_dim = emb_dim if not concat else emb_dim * num_layer
        self.shuffle_ratio = shuffle_ratio

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, concat=concat, last_act=True)

        self.weight = nn.Parameter(torch.empty(self.emd_dim, self.emd_dim))
        uniform(self.emd_dim, self.weight)

        self.projectoer = LinearPred(self.emd_dim, emb_dim, 2, 2, act='relu')
        

    def forward(self, x, edge_index, edge_weight=None, batch=None):

        pos_h = self.encoder(x, edge_index, edge_weight)

        x_cor = infomax_corruption(x, batch, self.shuffle_ratio)
        neg_h = self.encoder(x_cor, edge_index, edge_weight)

        # summary = torch.sigmoid(pos_h.mean(dim=0))
        summary = torch.sigmoid(global_mean_pool(pos_h, batch))

        return pos_h, neg_h, summary
    
    def discriminate(self, h, summary):

        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(h, torch.matmul(self.weight, summary))
        return torch.sigmoid(value)
    
    def get_loss(self, x, edge_index, edge_weight=None, batch=None):

        pos_h, neg_h, summary = self.forward(x, edge_index, edge_weight=edge_weight, batch=batch)
        if batch is None:
            pos_loss = -torch.log(self.discriminate(pos_h, summary) + self.EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(neg_h, summary) + self.EPS).mean()
            loss = pos_loss + neg_loss
        else:
            pos_loss = -torch.log(self.discriminate(pos_h, summary)[range(len(batch)), batch] + self.EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(neg_h, summary)[range(len(batch)), batch] + self.EPS).mean()
            loss = pos_loss + neg_loss

        return loss

    def embed(self, x, edge_index, edge_weight=None, batch=None):
        
        return self.encoder.embed(x, edge_index, edge_weight=edge_weight, batch=batch)

    def predict(self, x, edge_index, edge_weight=None, batch=None):
        return self.projectoer(self.encoder(x, edge_index, edge_weight=edge_weight, batch=batch))


class GraphCL(nn.Module):

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None, concat=False, 
                 tau=1, pooler='sum', contrast_batch=None):
        super().__init__()

        # self.lin_emb_dim = emb_dim * num_layer
        self.lin_emb_dim = emb_dim
        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, concat=False, pooler=pooler)
        self.proj_head = LinearPred(self.lin_emb_dim, emb_dim, emb_dim, 2)

        self.tau = tau
        self.contrast_batch = contrast_batch

    def forward(self, x, edge_index, edge_weight=None, batch=None):

        h = self.proj_head(self.encoder(x, edge_index, edge_weight, batch))
        return h
    
    def contrast_loss(self, x, x_aug, batch_size):

        n_samples, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        # Node/Graph level
        if batch_size:
            if n_samples <= batch_size:
                batch_size = n_samples
                sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
                sim_matrix = torch.exp(sim_matrix / self.tau)
                pos_sim = sim_matrix[range(batch_size), range(batch_size)]
                loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
                loss = -torch.log(loss).mean()
            else:
                n_loop = n_samples // batch_size + 1
                losses = []
                for i in range(n_loop):
                    start = i*batch_size
                    end = (i + 1)*batch_size if i != n_loop - 1 else n_samples
                    n_sim = batch_size if i != n_loop - 1 else end - start
                    sim_matrix = torch.einsum('ik,jk->ij', x[start:end], x_aug) / torch.einsum('i,j->ij', x_abs[start:end], x_aug_abs)
                    sim_matrix = torch.exp(sim_matrix / self.tau)
                    pos_sim = sim_matrix[range(n_sim), range(n_sim)]
                    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
                    losses.append(-torch.log(loss))
                
                loss = torch.concat(losses).mean()
        else:
            sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
            sim_matrix = torch.exp(sim_matrix / self.tau)
            pos_sim = sim_matrix[range(n_samples), range(n_samples)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = -torch.log(loss).mean()

        return loss

    def get_loss(self, x, edge_index, edge_weight=None, batch=None):

        if is_torch_sparse_tensor(edge_index):
            dense_edge_index, _ = to_edge_index(edge_index)
            x_aug, edge_index_aug = graphcl_augmentation(x, dense_edge_index)
            edge_index_aug = to_torch_sparse_tensor(edge_index_aug, size=x.size(0))
            del dense_edge_index
        else:
            x_aug, edge_index_aug = graphcl_augmentation(x, edge_index)

        h = self.forward(x, edge_index, edge_weight=edge_weight, batch=batch)
        h_aug = self.forward(x_aug, edge_index_aug, edge_weight=edge_weight, batch=batch)

        loss = self.contrast_loss(h, h_aug, self.contrast_batch)

        return loss
    
    def embed(self, x, edge_index, edge_weight=None, batch=None):

        return self.forward(x, edge_index, edge_weight=edge_weight, batch=batch)
    

class GraphGD(nn.Module):

    EPS = 1e-15

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None, concat=False):
        super().__init__()

        self.emd_dim = emb_dim

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, concat=False, last_act=True)
        self.proj_head = LinearPred(emb_dim, emb_dim, emb_dim, 1)
        

    def forward(self, x, edge_index, edge_weight=None, batch=None):
    
        pos_h = self.proj_head(self.encoder(x, edge_index, edge_weight, batch))

        x_cor = infomax_corruption(x, batch)
        neg_h = self.proj_head(self.encoder(x_cor, edge_index, edge_weight, batch))

        embed_h = pos_h.clone()
        # for _ in range(1):
        #     embed_h = edge_index.matmul(embed_h) + embed_h

        return embed_h, pos_h, neg_h
    
    def get_loss(self, pos_h, neg_h):

        pos_loss = -torch.log(torch.sigmoid(pos_h.sum(dim=1)) + self.EPS).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_h.sum(dim=1)) + self.EPS).mean()

        return pos_loss + neg_loss
    
    def embed(self, x, edge_index, edge_weight=None, batch=None):
        
        return self.proj_head(self.encoder(x, edge_index, edge_weight=edge_weight, batch=batch))


# ======================================================================
#   Predictive SSL
# ======================================================================

class GraphMAE(nn.Module):

    EPS = 1e-15

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None, concat=False, mask_ratio=0.5, replace_ratio=0):
        super().__init__()

        self.emb_dim = emb_dim if not concat else num_layer * emb_dim
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, 
                               concat=concat, last_act=True, aggr='mean')
        self.decoder = Encoder(emb_dim, in_dim, 1, kernel, drop_ratio, act, norm=None, 
                               concat=False, last_act=False, aggr='mean')

        self.encoder_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.encoder_to_decoder = nn.Linear(self.emb_dim, emb_dim, bias=False)


    def forward(self, x, edge_index, edge_weight=None, batch=None):
        # Mask
        mask_x, mask_nodes = self.encding_mask(x)
        h = self.encoder(mask_x, edge_index, edge_weight, batch)
        h = self.encoder_to_decoder(h)

        # Re-mask
        h[mask_nodes] = 0
        recon = self.decoder(h, edge_index, edge_weight, batch)

        return x[mask_nodes], recon[mask_nodes]


    def encding_mask(self, x):

        num_nodes = x.shape[0]

        # random masking
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self.mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        if self.replace_ratio > 0:
            num_noise_nodes = int(self.replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int((1 - self.replace_ratio) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_ratio * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.encoder_mask_token

        return out_x, mask_nodes
    

    def get_loss(self, x, edge_index, edge_weight=None, batch=None):

        mask_x, recon_x = self.forward(x, edge_index, edge_weight, batch)
        loss = sce_loss(mask_x, recon_x, alpha=2)

        return loss
    
    def embed(self, x, edge_index, edge_weight=None, batch=None):

        return self.encoder.embed(x, edge_index, edge_weight=edge_weight, batch=batch)


# ======================================================================
#   Fused SSL
# ======================================================================

class DGI_MAE(nn.Module):

    EPS = 1e-15

    def __init__(self, in_dim, emb_dim, num_layer, kernel='gcn', drop_ratio=0,
                 act='relu', norm=None, concat=False, mask_ratio=0.5, replace_ratio=0):
        super().__init__()

        self.emb_dim = emb_dim if not concat else num_layer * emb_dim
        self.mask_ratio = mask_ratio
        self.replace_ratio = replace_ratio

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, 
                               concat=concat, last_act=True, aggr='mean')
        self.decoder = Encoder(emb_dim, in_dim, 1, kernel, drop_ratio, act, norm=None, 
                               concat=False, last_act=False, aggr='mean')

        self.encoder_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.encoder_to_decoder = nn.Linear(self.emb_dim, emb_dim, bias=False)

        self.weight = nn.Parameter(torch.empty(emb_dim, emb_dim))
        uniform(emb_dim, self.weight)
    

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        
        # Contrastive Part
        pos_h = self.encoder(x, edge_index, edge_weight)
        summary = torch.sigmoid(global_mean_pool(pos_h, batch))
        x_cor = infomax_corruption(x, batch)
        neg_h = self.encoder(x_cor, edge_index, edge_weight)

        # Mask
        mask_x, mask_nodes = self.encding_mask(x)
        h = self.encoder(mask_x, edge_index, edge_weight)
        
        h = self.encoder_to_decoder(h)

        # Re-mask
        h[mask_nodes] = 0
        recon = self.decoder(h, edge_index, edge_weight)

        return pos_h, neg_h, summary, x[mask_nodes], recon[mask_nodes]
    
    def encding_mask(self, x):

        num_nodes = x.shape[0]

        # random masking
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self.mask_ratio * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        if self.replace_ratio > 0:
            num_noise_nodes = int(self.replace_ratio * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int((1 - self.replace_ratio) * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self.replace_ratio * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x_cl = out_x.clone()
        out_x[token_nodes] += self.encoder_mask_token

        return out_x, mask_nodes
    
    def discriminate(self, h, summary):

        summary = summary.t() if summary.dim() > 1 else summary
        value = torch.matmul(h, torch.matmul(self.weight, summary))
        return torch.sigmoid(value)
    
    def dgi_get_loss(self, pos_h, neg_h, summary, batch):

        if batch is None:
            pos_loss = -torch.log(self.discriminate(pos_h, summary) + self.EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(neg_h, summary) + self.EPS).mean()
            loss = pos_loss + neg_loss
        else:
            pos_loss = -torch.log(self.discriminate(pos_h, summary)[range(len(batch)), batch] + self.EPS).mean()
            neg_loss = -torch.log(1 - self.discriminate(neg_h, summary)[range(len(batch)), batch] + self.EPS).mean()
            loss = pos_loss + neg_loss

        return pos_loss + neg_loss
    
    
    def get_loss(self, x, edge_index, beta=1, edge_weight=None, batch=None):

        pos_h, neg_h, summary, mask_x, recon_x = self.forward(x, edge_index, edge_weight, batch)
        mae_loss_1 = sce_loss(mask_x, recon_x, alpha=2)
        dgi_loss = self.dgi_get_loss(pos_h, neg_h, summary, batch)

        loss = dgi_loss + beta*(mae_loss_1)

        return loss

    def embed(self, x, edge_index, edge_weight=None, batch=None):

        return self.encoder.embed(x, edge_index, edge_weight=edge_weight, batch=batch)


# ======================================================================
#   Non Parametric
# ======================================================================

class NonParametric(nn.Module):

    def __init__(self, in_dim, emb_dim, num_layer, kernel='simpleconv', drop_ratio=0,
                 act='relu', norm=None, concat=False):
        super().__init__()

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act, norm, concat=concat, last_act=True)

    def forward(self, x, edge_index):

        out = self.encoder(x, edge_index)
        return out
    
    def embed(self, x, edge_index):

        return self.encoder.embed(x, edge_index)


# ======================================================================
#   Supervised baseines
# ======================================================================
    
class GNN(nn.Module):

    def __init__(self, in_dim, emb_dim, num_layer=2, mlp_layer=1, kernel='gcn', drop_ratio=0,
                 act='relu', aggr='mean', norm='batchnorm'):
        super().__init__()

        self.encoder = Encoder(in_dim, emb_dim, num_layer, kernel, drop_ratio, act,
                               last_act=True, aggr=aggr, norm=norm)
        self.mlp = LinearPred(emb_dim, emb_dim, 2, mlp_layer, act=act)
    

    def forward(self, x, edge_index, edge_weight=None, batch=None):

        h = self.encoder(x, edge_index, edge_weight, batch)
        h = self.mlp(h)

        return h