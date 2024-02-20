import os
import shutil
import torch
import torch.nn.functional as F
import os.path as osp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from torch_geometric.datasets import DGraphFin, EllipticBitcoinDataset, HeterophilousGraphDataset, TUDataset
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import degree, from_scipy_sparse_matrix, coalesce
from dgl import load_graphs
from sklearn.model_selection import train_test_split
from scipy import io

from utils import set_random_seed


# ======================================================================
#   Global Variables
# ======================================================================

TUDATASET = [
    "MCF-7", "MOLT-4", "PC-3", "SW-620", "NCI-H23", "OVCAR-8", "P388", 
    "SF-295", "SN12C", "UACC257", 
    "Mutagenicity", "PROTEINS_full", "ENZYMES", "AIDS", "DHFR", "BZR", 
    "COX2", "DD", "NCI1", "IMDB-BINARY", "IMDB-MULTI", "KKI", "OSHU",
    "Tox21_HSE_training", "Tox21_HSE_testing", "REDDIT-BINARY"
]

MOL = ["MCF-7", "MOLT-4", "PC-3", "SW-620", "NCI-H23", "OVCAR-8", "P388", 
       "SF-295", "SN12C", "UACC257"]

NODEDATASET = [
    'yelp', 'amazon', 'weibo', 'reddit', 'tfinance', 'tsocial', 'elliptic',
    'dgraphfin', 'questions', 'tolokers'
]
SYNTHETIC = ['inj_cora', 'inj_flickr', 'inj_amazon', 'acm', 'blogcatalog']
NODEDATASET += SYNTHETIC


# ======================================================================
#   Node-level Anomaly Detection Dataset
# ======================================================================

class FraudDataset(InMemoryDataset):
    '''
    Fraud dataset (YelpChi & Amazon), code borrowed from DGL
    '''
    url = 'https://data.dgl.ai/'
    file_urls = {
        "yelp": "dataset/FraudYelp.zip",
        "amazon": "dataset/FraudAmazon.zip",
    }
    relations = {
        "yelp": ["net_rsr", "net_rtr", "net_rur"],
        "amazon": ["net_upu", "net_usu", "net_uvu"],
    }
    file_names = {"yelp": "YelpChi.mat", "amazon": "Amazon.mat"}
    node_name = {"yelp": "review", "amazon": "user"}

    def __init__(self, root, name, transform=None, pre_transform=None, random_seed=717, 
                 train_size=0.7, val_size=0.1, force_reload=False):

        self.name = name
        assert self.name in ['yelp', 'amazon']

        self.url = osp.join(self.url, self.file_urls[self.name])
        self.seed = random_seed
        self.train_size = train_size
        self.val_size = val_size

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        names = [self.file_names[self.name]]
        return names
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
    
    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        data = io.loadmat(file_path)
        node_features = torch.FloatTensor(data["features"].todense())
        # remove additional dimension of length 1 in raw .mat file
        node_labels = torch.LongTensor(data["label"].squeeze())
        edge_index = []
        for relation in self.relations[self.name]:
            edge_index.append(from_scipy_sparse_matrix(data[relation].tocoo())[0])
        edge_index = coalesce(torch.concat(edge_index, dim=1))

        data = Data(x=node_features, edge_index=edge_index, y=node_labels)

        data = self._random_split(data, self.seed, self.train_size, self.val_size)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
        
    def _random_split(self, data, seed=717, train_size=0.7, val_size=0.1):
        """split the dataset into training set, validation set and testing set"""

        assert 0 <= train_size + val_size <= 1, (
            "The sum of valid training set size and validation set size "
            "must between 0 and 1 (inclusive)."
        )

        N = data.x.shape[0]
        index = np.arange(N)
        if self.name == "amazon":
            # 0-3304 are unlabeled nodes
            index = np.arange(3305, N)

        index = np.random.RandomState(seed).permutation(index)
        train_idx = index[: int(train_size * len(index))]
        val_idx = index[len(index) - int(val_size * len(index)) :]
        test_idx = index[
            int(train_size * len(index)) : len(index)
            - int(val_size * len(index))
        ]
        train_mask = np.zeros(N, dtype=np.bool_)
        val_mask = np.zeros(N, dtype=np.bool_)
        test_mask = np.zeros(N, dtype=np.bool_)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        data.train_mask = torch.tensor(train_mask)
        data.val_mask = torch.tensor(val_mask)
        data.test_mask = torch.tensor(test_mask)

        return data

    def __repr__(self):
        return f'{self.name}()'


class TDataset(InMemoryDataset):
    '''
    Tencent dataset (T-Finance & T-Social), code borrowed from GADBench
    '''
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        assert self.name in ['tfinance', 'tsocial']

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        names = [self.name]
        return names
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        pass

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        if not osp.exists(file_path):
            try:
                shutil.copy(f'data/{self.name}', file_path)
            except:
                raise ValueError('Source data does not exist!\n\
                                 Please download the source data from GADBench to ./data/{name}.\n\
                                 GitHub link: https://github.com/squareRoot3/GADBench')
        
        data = load_graphs(file_path)[0][0]
        features = data.ndata['feature']
        labels = data.ndata['label']
        train_mask = data.ndata['train_masks']
        val_mask = data.ndata['val_masks']
        test_mask = data.ndata['test_masks']
        
        data = Data(x=features, edge_index=torch.vstack(data.edges()), y=labels)
        data.tran_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


class TextDataset(InMemoryDataset):
    '''
    Text dataset, code borrowed from GADBench
    '''
    url = 'https://github.com/pygod-team/data/raw/main/'
    file_urls = {
        "reddit": "reddit.pt.zip",
        "weibo": "weibo.pt.zip",
        "inj_cora": "inj_cora.pt.zip",
        "inj_flickr": "inj_flickr.pt.zip",
        "inj_amazon": "inj_amazon.pt.zip"
    }
    file_names = {
        "reddit": "reddit.pt", "weibo": "weibo.pt",
        "inj_cora": "inj_cora.pt", "inj_flickr": "inj_flickr.pt", 
        "inj_amazon": "inj_amazon.pt" 
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        
        self.name = name
        assert self.name in ['reddit', 'weibo', 'inj_cora', 'inj_flickr', "inj_amazon"]

        self.url = osp.join(self.url, self.file_urls[self.name])

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        names = [self.file_names[self.name]]
        return names
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = torch.load(file_path)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])


class SynDataset(InMemoryDataset):
    '''
    Synthetic anomaly dataset, code borrowed from BOND
    '''
    # url = 'https://github.com/mala-lab/TAM-master/raw/main/data'
    url = {
        "acm": 'https://github.com/GRAND-Lab/CoLA/raw/main/dataset/',
        "blogcatalog": 'https://github.com/mala-lab/TAM-master/raw/main/data'
    }
    file_urls = {
        "acm": "ACM.mat",
        "blogcatalog": "BlogCatalog.mat",
    }
    file_names = {
        "acm": "ACM.mat",
        "blogcatalog": "BlogCatalog.mat",
    }

    def __init__(self, root, name, transform=None, pre_transform=None):
        
        self.name = name
        assert self.name in ['acm', 'blogcatalog']

        # self.url = osp.join(self.url, self.file_urls[self.name])
        self.url = osp.join(self.url[self.name], self.file_urls[self.name])

        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        names = [self.file_names[self.name]]
        return names
    
    @property
    def processed_file_names(self):
        return 'data.pt'
    
    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = io.loadmat(file_path)

        label = data['Label'] if ('Label' in data) else data['gnd']
        attr = data['Attributes'] if ('Attributes' in data) else data['X']
        network = data['Network'] if ('Network' in data) else data['A']

        edge_index = from_scipy_sparse_matrix(sp.csr_matrix(network).tocoo())[0]
        edge_index = coalesce(edge_index)
        features = torch.FloatTensor(attr.todense())
        ano_labels = torch.tensor(label).squeeze()
        if 'str_anomaly_label' in data:
            str_ano_labels = torch.tensor(data['str_anomaly_label']).squeeze()
            attr_ano_labels = torch.tensor(data['attr_anomaly_label']).squeeze()
            dy = torch.zeros_like(ano_labels)
            dy[attr_ano_labels == 1] = 1
            dy[str_ano_labels == 1] = 2
        else:
            str_ano_labels = None
            attr_ano_labels = None
            dy = None

        data = Data(x=features, edge_index=edge_index, y=ano_labels)
        if dy is not None:
            data.dy = dy
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])



# ======================================================================
#   Graph-level Anomaly Detection Dataset Loader
# ======================================================================

def load_tu_dataset(root, dataset_name):
    '''
    TUDataset, code borrowed from previous works
    '''
    dataset = TUDataset(root, dataset_name, use_node_attr=True)

    if dataset.num_node_labels > 0 and not dataset.name in ['MUTAG']:
        print('Use node labels as attributes.')
    else:
        print('Use node degrees as attributes.')

        if not hasattr(dataset._data, '_num_nodes'):
            dataset._data._num_nodes = dataset.slices['x'].diff()

        # Calculate node degrees
        feat_dim, MAX_DEGREE = 0, 400
        degrees = []
        x_slices = [0]
        for i in range(dataset.len()):
            start, end = dataset.slices['edge_index'][i:(i+2)].tolist()
            num_nodes = dataset._data._num_nodes[i]
            cur_degree = degree(dataset._data.edge_index[0, start:end], num_nodes).long()
            feat_dim = max(feat_dim, cur_degree.max().item())
            degrees.append(cur_degree)
            x_slices.append(x_slices[-1] + num_nodes)

        degrees = torch.cat(degrees)

        # Restrict maximum degree
        feat_dim = min(feat_dim, MAX_DEGREE) + 1
        degrees[degrees > MAX_DEGREE] = MAX_DEGREE
        x = F.one_hot(degrees, num_classes=feat_dim).float()

        # Record
        dataset._data.x = x
        dataset.slices['x'] = torch.tensor(x_slices, dtype=dataset.slices['edge_index'].dtype)

    if dataset_name in MOL:
        dataset = mol_split(dataset)
    else:
        dataset = tu_split(dataset)

    return dataset


def mol_split(dataset):

    # Semi-supervised split
    labels = dataset.y
    n_samples = labels.shape[0]
    index = list(range(n_samples))

    samples = 20
    semi_train_masks = torch.zeros([n_samples, 5]).bool()
    semi_val_masks = torch.zeros([n_samples, 5]).bool()
    semi_test_masks = torch.zeros([n_samples, 5]).bool()
    pos_index = np.where(labels == 1)[0]
    neg_index = list(set(index) - set(pos_index))
    for i in range(5):
        set_random_seed(i)
        pos_train_idx = np.random.choice(pos_index, size=2*samples, replace=False)
        set_random_seed(i)
        neg_train_idx = np.random.choice(neg_index, size=8*samples, replace=False)
        train_idx = np.concatenate([pos_train_idx[:samples], neg_train_idx[:4*samples]])
        semi_train_masks[train_idx, i] = 1
        val_idx = np.concatenate([pos_train_idx[samples:], neg_train_idx[4*samples:]])
        semi_val_masks[val_idx, i] = 1
        semi_test_masks[index, i] = 1
        semi_test_masks[train_idx, i] = 0
        semi_test_masks[val_idx, i] = 0

    dataset.semi_train_masks = semi_train_masks
    dataset.semi_val_masks = semi_val_masks
    dataset.semi_test_masks = semi_test_masks

    # Supervised split
    train_ratio, val_ratio = 0.7, 0.15
    train_masks = torch.zeros([n_samples, 5]).bool()
    val_masks = torch.zeros([n_samples, 5]).bool()
    test_masks = torch.zeros([n_samples, 5]).bool()
    for i in range(5):
        seed = 3407 + 10*i
        set_random_seed(seed)
        idx_train, idx_rest, _, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=train_ratio, random_state=seed, shuffle=True)
        idx_valid, idx_test, _, _ = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=int(len(index)*val_ratio), random_state=seed, shuffle=True)
        train_masks[idx_train, i] = 1
        val_masks[idx_valid, i] = 1
        test_masks[idx_test, i] = 1
    
    dataset.train_masks = train_masks
    dataset.val_masks = val_masks
    dataset.test_masks = test_masks

    return dataset


def tu_split(dataset):

    # Reverse labels
    new_y = dataset.y
    if len(dataset.y.unique()) > 2:
        new_y[new_y != 0] = 1
    new_y = 1 - new_y
    dataset._data.y = new_y

    # Downsample
    labels = dataset.y
    pos_index = np.where(labels == 1)[0]
    neg_index = np.where(labels == 0)[0]
    set_random_seed(0)
    pos_sample_idx = np.random.choice(pos_index, size=int(0.1*len(pos_index)), replace=False)
    index = np.concatenate([pos_sample_idx, neg_index])
    dataset = dataset[index]
    
    # Split
    labels = dataset.y
    n_samples = dataset.y.shape[0]
    index = np.arange(n_samples)

    # Semi-supervised split
    semi_train_masks = torch.zeros([n_samples, 5]).bool()
    semi_val_masks = torch.zeros([n_samples, 5]).bool()
    semi_test_masks = torch.zeros([n_samples, 5]).bool()
    for i in range(5):
        set_random_seed(i)
        idx_train, idx_rest, _, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=0.05, random_state=i, shuffle=True)
        idx_valid, idx_test, _, _ = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=int(len(index)*0.05), random_state=i, shuffle=True)
        semi_train_masks[idx_train, i] = 1
        semi_val_masks[idx_valid, i] = 1
        semi_test_masks[idx_test, i] = 1
    
    dataset.semi_train_masks = semi_train_masks
    dataset.semi_val_masks = semi_val_masks
    dataset.semi_test_masks = semi_test_masks

    # Supervised split
    train_ratio, val_ratio = 0.7, 0.15
    train_masks = torch.zeros([n_samples, 5]).bool()
    val_masks = torch.zeros([n_samples, 5]).bool()
    test_masks = torch.zeros([n_samples, 5]).bool()
    for i in range(5):
        seed = 3407 + 10*i
        set_random_seed(seed)
        idx_train, idx_rest, _, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=train_ratio, random_state=seed, shuffle=True)
        idx_valid, idx_test, _, _ = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=int(len(index)*val_ratio), random_state=seed, shuffle=True)
        train_masks[idx_train, i] = 1
        val_masks[idx_valid, i] = 1
        test_masks[idx_test, i] = 1
    
    dataset.train_masks = train_masks
    dataset.val_masks = val_masks
    dataset.test_masks = test_masks

    return dataset


# ======================================================================
#   Node-level Anomaly Detection Dataset Loader
# ======================================================================

def load_node_dataset(root, name, samples, supervised):
    """
    Wrapper functions for node level data loader
    """

    if name in ['yelp', 'amazon']:
        dataset = FraudDataset(root, name)
    elif name in ['weibo', 'reddit', 'inj_cora', 'inj_flickr', 'inj_amazon']:
        dataset = TextDataset(root, name)
    elif name in ['tfinance', 'tsocial']:
        dataset = TDataset(root, name)
    elif name == 'elliptic':
        dataset = EllipticBitcoinDataset(osp.join(root, name))
        timestep = pd.read_csv(dataset.raw_paths[0], header=None).iloc[:, 1]
        timestep = torch.tensor(timestep, dtype=dataset.x.dtype)
        dataset.x = torch.concat([timestep.unsqueeze(dim=0).T, dataset.x], dim=1)
    elif name == 'dgraphfin':
        dataset = DGraphFin(osp.join(root, name))
    elif name in ['questions', 'tolokers']:
        dataset = HeterophilousGraphDataset(root, name.capitalize())
    elif name in ['acm', 'blogcatalog']:
        dataset = SynDataset(root, name)

    # Check if preprocess needed
    if name in ['amazon', 'yelp']:
        x = dataset.x
        x = (x - x.mean(0)) / x.std(0)
        dataset.x = x
    
    if name in ['inj_cora', 'inj_flickr', 'inj_amazon']:
        labels = dataset.y
        dataset.dy = labels.clone()
        labels[labels != 0] = 1
        dataset.y = labels

    # Valid index
    labels = dataset.y
    n_nodes = dataset.x.shape[0]
    if hasattr(dataset, 'train_mask'):
        if hasattr(dataset, 'val_mask'):
            index = (dataset.train_mask | dataset.val_mask | dataset.test_mask).nonzero()[:, 0].numpy().tolist()
        else:
            index = (dataset.train_mask | dataset.test_mask).nonzero()[:, 0].numpy().tolist()
    else:
        index = list(range(n_nodes))


    file_path = osp.join('data', name)
    if not osp.exists(file_path) and not name in SYNTHETIC:
        print(f'Recommended splits do not exist! Randomized splits are used for this run.\n\
              To better repoduce the results, download the data from GADBench to ./data/{name} and use its split.\n\
              GitHub link: https://github.com/squareRoot3/GADBench')

    # Supervised split
    if osp.exists(file_path):
        bench_data = load_graphs(file_path)[0][0]

        dataset.train_masks = bench_data.ndata['train_masks'][:, :10].bool()
        dataset.val_masks = bench_data.ndata['val_masks'][:, :10].bool()
        dataset.test_masks = bench_data.ndata['test_masks'][:, :10].bool()
    else:
        if name in ['tolokers', 'questions']:
            train_ratio, val_ratio = 0.5, 0.25
        elif name in ['tsocial', 'tfinance', 'reddit', 'weibo']:
            train_ratio, val_ratio = 0.4, 0.2
        
        if name in ['amazon', 'yelp', 'dgraphfin']: # official split
            dataset.train_masks = dataset.train_mask.repeat(10,1).T
            dataset.val_masks = dataset.val_mask.repeat(10,1).T
            dataset.test_masks = dataset.test_mask.repeat(10,1).T
        elif name == 'elliptic':
            train_mask = (dataset.train_mask) & (dataset.x[:, 0] <= 25)
            val_mask = (dataset.train_mask) & (dataset.x[:, 0] > 25) & (dataset.x[:, 0] <= 34)
            dataset.train_masks = train_mask.repeat(10,1).T
            dataset.val_masks = val_mask.repeat(10,1).T
            dataset.test_masks = dataset.test_mask.repeat(10,1).T
        else:
            train_masks = torch.zeros([n_nodes, 10]).bool()
            val_masks = torch.zeros([n_nodes, 10]).bool()
            test_masks = torch.zeros([n_nodes, 10]).bool()
            for i in range(10):
                seed = 3407 + 10*i
                set_random_seed(seed)
                idx_train, idx_rest, _, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=train_ratio, random_state=seed, shuffle=True)
                idx_valid, idx_test, _, _ = train_test_split(idx_rest, y_rest, stratify=y_rest, train_size=int(len(index)*val_ratio), random_state=seed, shuffle=True)
                train_masks[idx_train, i] = 1
                val_masks[idx_valid, i] = 1
                test_masks[idx_test, i] = 1
            
            dataset.train_masks = train_masks
            dataset.val_masks = val_masks
            dataset.test_masks = test_masks

    if name in SYNTHETIC:
        dataset.train_masks = torch.zeros([n_nodes, 10]).bool()
        dataset.val_masks = torch.zeros([n_nodes, 10]).bool()
        dataset.test_masks = torch.zeros([n_nodes, 10]).bool()

    # Semi-supervised split
    if samples == 0:
        if osp.exists(file_path):
            dataset.semi_train_masks = bench_data.ndata['train_masks'][:, 10:].bool()
            dataset.semi_val_masks = bench_data.ndata['val_masks'][:, 10:].bool()
            dataset.semi_test_masks = bench_data.ndata['test_masks'][:, 10:].bool()
        else:
            samples = 20
            semi_train_masks = torch.zeros([n_nodes, 10]).bool()
            semi_val_masks = torch.zeros([n_nodes, 10]).bool()
            semi_test_masks = torch.zeros([n_nodes, 10]).bool()
            for i in range(10):
                pos_index = np.where(labels == 1)[0]
                neg_index = list(set(index) - set(pos_index))
                set_random_seed(i)
                pos_train_idx = np.random.choice(pos_index, size=2*samples, replace=False)
                set_random_seed(i)
                neg_train_idx = np.random.choice(neg_index, size=8*samples, replace=False)
                train_idx = np.concatenate([pos_train_idx[:samples], neg_train_idx[:4*samples]])
                semi_train_masks[train_idx, i] = 1
                val_idx = np.concatenate([pos_train_idx[samples:], neg_train_idx[4*samples:]])
                semi_val_masks[val_idx, i] = 1
                semi_test_masks[index, i] = 1
                semi_test_masks[train_idx, i] = 0
                semi_test_masks[val_idx, i] = 0

            dataset.semi_train_masks = semi_train_masks
            dataset.semi_val_masks = semi_val_masks
            dataset.semi_test_masks = semi_test_masks
    else:
        semi_train_masks = torch.zeros([n_nodes, 10]).bool()
        semi_val_masks = torch.zeros([n_nodes, 10]).bool()
        semi_test_masks = torch.zeros([n_nodes, 10]).bool()
        for i in range(10):
            pos_index = np.where(labels == 1)[0]
            neg_index = list(set(index) - set(pos_index))
            set_random_seed(i)
            pos_train_idx = np.random.choice(pos_index, size=2*samples, replace=False)
            set_random_seed(i)
            neg_train_idx = np.random.choice(neg_index, size=8*samples, replace=False)
            train_idx = np.concatenate([pos_train_idx[:samples], neg_train_idx[:4*samples]])
            semi_train_masks[train_idx, i] = 1
            val_idx = np.concatenate([pos_train_idx[samples:], neg_train_idx[4*samples:]])
            semi_val_masks[val_idx, i] = 1
            semi_test_masks[index, i] = 1
            semi_test_masks[train_idx, i] = 0
            semi_test_masks[val_idx, i] = 0

        dataset.semi_train_masks = semi_train_masks
        dataset.semi_val_masks = semi_val_masks
        dataset.semi_test_masks = semi_test_masks

    return dataset

# ======================================================================
#   Load dataset
# ======================================================================

def load_dataset(root, dataset, samples=0, supervised=False):
    """
    Wrapper functions for data loader
    """

    if dataset in NODEDATASET:
        dataset = load_node_dataset(root, dataset, samples, supervised)
    elif dataset in TUDATASET:
        dataset = load_tu_dataset(root, dataset)
    
    return dataset


if __name__ == '__main__':

    # dataset = FraudDataset(root='dataset', name='amazon')
    # dataset = TextDataset(root='dataset', name='weibo')
    dataset = TDataset(root='dataset', name='tsocial')
    print('ok')