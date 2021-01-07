import numpy as np
import torch
from torch.utils import data
import pandas as pd

class PPI(data.Dataset):
    # df : a list of data, which includes an index for the pair, an index for entity1 and entity2, from a list that combines all the entities. we want the
    def __init__(self, df):
        'Initialization'
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        idx1 = self.df.iloc[index,0]
        idx2 = self.df.iloc[index,1]
        y = self.df.iloc[index,2]
        return y, (idx1, idx2), index

def get_node_mapping(nodes):
    r"""get mapping functions for each node
    """
    prot_to_idx = {x:i for i,x in enumerate(nodes)}
    idx_to_prot = {i:x for i,x in enumerate(nodes)}
    prot_to_idx = np.vectorize(prot_to_idx.get)
    idx_to_prot = np.vectorize(idx_to_prot.get)
    return prot_to_idx, idx_to_prot

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
