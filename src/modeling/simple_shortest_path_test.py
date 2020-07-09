import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import train_test_split_edges
from datasets.Interactomes import Interactomes
from utils.train_test_split_same_dist import train_test_split_same_dist
from utils.train_test_split_same_dist import get_shortest_path
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

torch.manual_seed(12345)

dataset = Interactomes('../../Data/interactomes/human/', 'human')
data = dataset[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_same_dist(data, val_ratio=0.05, test_ratio=0.05)

x, pos_edge_index = data.x, data.train_pos_edge_index

_edge_index, _ = remove_self_loops(pos_edge_index)
pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                   num_nodes=x.size(0))

neg_edge_index = negative_sampling(
    edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
    num_neg_samples=pos_edge_index.size(1))

print(neg_edge_index.size())
print(pos_edge_index.size())
print(data.val_pos_edge_index.size())
print(data.val_neg_edge_index.size())
print(data.test_pos_edge_index.size())
print(data.test_neg_edge_index.size())

train_neg = [tuple(sorted(x)) for x in neg_edge_index.cpu().numpy().T]
train_pos = [tuple(sorted(x)) for x in pos_edge_index.cpu().numpy().T]
val_pos = [tuple(sorted(x)) for x in data.val_pos_edge_index.cpu().numpy().T]
val_neg = [tuple(sorted(x)) for x in data.val_neg_edge_index.cpu().numpy().T]
test_pos = [tuple(sorted(x)) for x in data.test_pos_edge_index.cpu().numpy().T]
test_neg = [tuple(sorted(x)) for x in data.test_neg_edge_index.cpu().numpy().T]

print(len(set(train_pos)))
print(len(set(train_neg)))
print()
print(len(set(val_pos)))
print(len(set(val_pos)))
print()
print(len(set(test_pos)))
print(len(set(test_neg)))

# print(pos_edge_index.size())
# print(data.val_pos_edge_index.size())
# print(data.val_neg_edge_index.size())
# print(data.test_pos_edge_index.size())
# print(data.test_neg_edge_index.size())

print(set(val_neg).intersection(set(train_neg)))
print(set(test_neg).intersection(set(train_neg)))
print(set(val_neg).intersection(set(test_neg)))
print()
print(set(test_pos+train_pos+val_pos).intersection(set(test_neg+train_neg+val_neg)))
