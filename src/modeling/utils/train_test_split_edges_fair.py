import math
import random
import torch
from torch_geometric.utils import to_undirected
from collections import Counter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def shortest_path_cutoff(train, edges, cutoff, size):

    G = nx.Graph()
    G.add_edges_from(train.cpu().numpy().T)
    np_edges = edges.cpu().numpy().T
    print(size)
    allowed_edges = []
    sp_len_total = []
    while len(set(allowed_edges)) < size:

        prev_size = len(set(allowed_edges))
        perm = random.sample(range(len(np_edges)), size)

        for i in perm:
            try:
                sp_length = nx.shortest_path_length(G, np_edges[i][0], np_edges[i][1])
            except:
                continue

            if sp_length > cutoff:
                continue
            if sp_length <= 1:
                continue

            allowed_edges.append(i)
            sp_len_total.append(sp_length)

        if len(set(allowed_edges)) == prev_size:
            break

    print(sum(i for x,i in set(zip(allowed_edges, sp_len_total)))/len(set(zip(allowed_edges, sp_len_total))))
    print(edges[:,np.array(list(set(allowed_edges)))][:,:size].size())
    return edges[:,np.array(list(set(allowed_edges)))][:,:size]

def train_test_split_edges_fair(data, val_ratio=0.05, test_ratio=0.1, cutoff=None):
    r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
    into positive and negative train/val/test edges, and adds attributes of
    `train_pos_edge_index`, `train_neg_adj_mask`, `val_pos_edge_index`,
    `val_neg_edge_index`, `test_pos_edge_index`, and `test_neg_edge_index`
    to :attr:`data`.
    Args:
        data (Data): The data object.
        val_ratio (float, optional): The ratio of positive validation
            edges. (default: :obj:`0.05`)
        test_ratio (float, optional): The ratio of positive test
            edges. (default: :obj:`0.1`)
    :rtype: :class:`torch_geometric.data.Data`
    """

    test_data = data.edge_index.cpu().numpy().T

    assert 'batch' not in data  # No batch-mode.

    num_nodes = data.num_nodes
    row, col = data.edge_index
    edges = data.edge_index
    data.edge_index = None

    # Return upper triangular portion.
    mask = row > col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

    if cutoff:
        data.val_pos_edge_index = shortest_path_cutoff(data.train_pos_edge_index, data.val_pos_edge_index, cutoff, data.val_pos_edge_index.size(1))
        n_v = data.val_pos_edge_index.size(1)
        data.test_pos_edge_index = shortest_path_cutoff(data.train_pos_edge_index, data.test_pos_edge_index, cutoff, data.test_pos_edge_index.size(1))
        n_t = data.test_pos_edge_index.size(1)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    if cutoff:
        negatives = neg_adj_mask.nonzero().T
        size = min(n_v + n_t, negatives.size(1))
        negs = shortest_path_cutoff(data.train_pos_edge_index, negatives, cutoff, size)
        neg_row, neg_col = negs
    else:
        neg_row, neg_col = neg_adj_mask.nonzero().T
        perm = random.sample(range(neg_row.size(0)),
                         min(n_v + n_t, neg_row.size(0)))
        perm = torch.tensor(perm)
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
