# Graph Pre-Training Models Are Strong Anomaly Detectors

Quick Start
----------------------

To reproduce the reported results, please run the script with `--use_cfg`.


**Node-level Graph Anomaly Detection**

```
# With best configurations
python pretrain_node.py --dataset <dataset> --use_cfg
```

Supported datasets includes `weibo`, `reddit`, `amazon`, `yelp`, `tfinance`, `elliptic`, `tolokers`, `questions`. `dgraphfin`, `tsocial`


**Graph-level Graph Anomaly Detection**

```
# With best configurations
python pretrain_graph.py --dataset <dataset> --use_cfg
```

Supported datasets includes `DD`, `IMDB-BINARY`, `REDDIT-BINARY`, `PROTEINS_full`, `AIDS`, `NCI1`, `Mutagenicity`