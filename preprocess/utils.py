import numpy as np
import math
import pandas as pd

np.random.seed(123)

def undirected_edges(edges):
    r"""Create an undirected network
    """
    flipped_edges = np.flip(edges, axis=1)
    edges = np.concatenate([edges, flipped_edges])
    edges = np.unique(edges, axis=0)
    return edges

def remove_self_loop(edges):
    r"""Remove self loops
    """
    return edges[edges[:,0] != edges[:,1]]

def train_test_split(edges, val_ratio=0.05, test_ratio=0.1):
    r"""Splits the edges of into positive and negative train/val/test edges
    """
    # train ratio not lower or equal to 0
    assert (1-val_ratio-test_ratio) >= 0

    row, col = edges[:,0], edges[:,1]

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.shape[0]))
    n_t = int(math.floor(test_ratio * row.shape[0]))

    # Shuffle
    perm = np.random.permutation(row.shape[0])
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    val = np.stack([r, c], axis=0).T
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    test = np.stack([r, c], axis=0).T

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    train = np.stack([r, c], axis=0).T
    train = undirected_edges(train)

    return train, val, test

def negatives(edges, edge_count = None):
    r"""generate top triangle negative edges
    """

    A = np.zeros((np.max(edges)+1, np.max(edges)+1))
    for x in edges:
        A[x[0], x[1]] = 1

    # Return upper triangular portion.
    row, col = np.where(A == 0)
    mask = row < col
    row, col = row[mask], col[mask]

    # shuffle
    perm = np.random.permutation(row.size)
    row, col = row[perm], col[perm]
    negatives = np.stack([row, col]).T

    if edge_count:
        return negatives[:edge_count]

    return negatives

def similar_proteins(train, all, mmseqs_file, seq_id=0.4):
    r"""Retrieve all protein similar to train proteins
    """

    align_df = pd.read_csv(mmseqs_file, sep='\t', header=None)
    align_df_sim = align_df[align_df[2] >= seq_id]
    align_df_sim = align_df_sim.iloc[:,:2]

    #only proteins from all allowed
    sim_filter = align_df_sim.isin(all).all(axis=1)
    align_df_sim = align_df_sim[sim_filter]

    # train sim pairs
    sim_train_filter = align_df_sim.isin(train).all(axis=1)
    align_df_sim = align_df_sim[sim_train_filter]

    train_sim_nodes = list(align_df_sim.values.flatten())
    train_sim_nodes += list(train)
    return np.unique(train_sim_nodes)

def c1_c2_c3(train_nodes_sim, edges):
    r"""Create C1, C2, C3 test sets
    """

    isin = np.isin(edges, train_nodes_sim)
    isin_sum = np.sum(isin, axis=1)

    c1 = edges[isin_sum == 2]
    c2 = edges[isin_sum == 1]
    c3 = edges[isin_sum == 0]
    return c1, c2, c3

def write_set(positives, negatives, path):
    r"""write positives and negatives to a csv file
    """
    df = pd.DataFrame(np.concatenate((positives, negatives)))
    df['label'] = np.array([1]*positives.shape[0] + [0]*negatives.shape[0])
    df.to_csv(path, index=False)
