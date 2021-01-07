import numpy as np
import pandas as pd

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
