import random

import torch
import numpy as np
import torch
from typing import Optional
import networkx as nx
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils import get_laplacian
import scipy.sparse as sp


def maybe_num_nodes(index: torch.Tensor,
                    num_nodes: Optional[int] = None) -> int:
    return int(index.max()) + 1 if num_nodes is None else num_nodes

def sample(high: int, size: int, device=None):
    size = min(high, size)
    return torch.tensor(random.sample(range(high), size), device=device)

def total_neighbors(G, x, y):
    return len(list(G.neighbors(x))) + len(list(G.neighbors(y)))

# https://link.springer.com/article/10.1007/s10115-014-0789-0
def negatives_sampling_above_distance(edge_index, train_edges, num_nodes=None, num_neg_samples=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    num_neg_samples = num_neg_samples or edge_index.size(1)

    # Handle '|V|^2 - |E| < |E|'.
    size = num_nodes * num_nodes
    num_neg_samples = min(num_neg_samples, size - edge_index.size(1))

    row, col = edge_index

    idx = row * num_nodes + col

    G = nx.Graph()
    G.add_nodes_from(list(range(num_nodes)))
    G.add_edges_from(train_edges.to('cpu').numpy().T)
    adj_norm = nx.normalized_laplacian_matrix(G)
    adj_orig_norm = adj_norm - sp.dia_matrix((adj_norm.diagonal()[np.newaxis, :], [0]), shape=adj_norm.shape)
    adj_orig_norm.eliminate_zeros()

    weights = adj_orig_norm.toarray().flatten()

    print(sum())

    # print(np.array([total_neighbors(G, x[0], x[1]) for x in train_edges.to('cpu').numpy().T]))
    # total_neighbors = []
    # for x in neg_edge_index.to('cpu').numpy().T:
    #     total_neighbors.append(len(list(G.neighbors(x[0])) + list(G.neighbors(x[1]))))

     # Percentage of edges to oversample so that we are save to only sample once
    # (in most cases).
    alpha = 1 / (1 - 1.1 * (edge_index.size(1) / size))
    perm = sample(size, int(alpha * num_neg_samples))
    mask = torch.from_numpy(np.isin(perm, idx.to('cpu'))).to(torch.bool)
    perm = perm[~mask][:num_neg_samples].to(edge_index.device)

    row = perm / num_nodes
    col = perm % num_nodes
    neg_edge_index = torch.stack([row, col], dim=0)

    # lengths = np.array(total_neighbors) > 30

    return neg_edge_index
