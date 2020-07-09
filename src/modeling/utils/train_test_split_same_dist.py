import math
import random
import torch
from torch_geometric.utils import to_undirected
from collections import Counter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def get_shortest_path(train, edges):
    G = nx.Graph()
    G.add_edges_from(train.cpu().numpy().T)
    np_edges = edges.cpu().numpy().T

    sp_edges = []
    for edge in np_edges:
        try:
            sp_length = nx.shortest_path_length(G, edge[0], edge[1])
        except:
            sp_length = 0

        sp_edges.append(sp_length)

    return sp_edges

def equal_neg_shortest_path_dist(train, edges, shortest_path_pos):
    G = nx.Graph()
    G.add_edges_from(train.cpu().numpy().T)
    np_edges = edges.cpu().numpy().T

    max_allowed = dict(Counter(shortest_path_pos))

    allowed_edges = []

    current_dist = {k:0 for k in max_allowed}

    while len(set(allowed_edges)) < len(shortest_path_pos):
        prev_size = len(set(allowed_edges))
        perm = random.sample(range(len(np_edges)), len(shortest_path_pos))

        for i in perm:
            try:
                sp_length = nx.shortest_path_length(G, np_edges[i][0], np_edges[i][1])
            except:
                sp_length = 0

            # if i in allowed_edges:
            #     continue
            if sp_length not in current_dist.keys():
                continue
            if current_dist[sp_length] >= max_allowed[sp_length]:
                continue
            if sp_length == 1:
                continue

            allowed_edges.append(i)
            current_dist[sp_length] += 1

        if len(set(allowed_edges)) == prev_size:
            print('ja')
            break

        print(current_dist)

    print(edges[:,np.array(list(set(allowed_edges)))].size())
    return edges[:,np.array(list(set(allowed_edges)))]

def train_test_split_same_dist(data, val_ratio=0.05, test_ratio=0.1):
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
    data.train_network = to_undirected(data.train_pos_edge_index)

    val_sp = get_shortest_path(data.train_pos_edge_index, data.val_pos_edge_index)
    test_sp = get_shortest_path(data.train_pos_edge_index, data.test_pos_edge_index)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    negatives = neg_adj_mask.nonzero().T

    negs = equal_neg_shortest_path_dist(data.train_pos_edge_index, negatives, val_sp+test_sp)
    neg_row, neg_col = negs

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data
