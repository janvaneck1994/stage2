import numpy as np
import os
import pandas as pd
import h5py
from Bio import SeqIO
from utils import *

np.random.seed(123)

output_path = '../data/processed/'
mmseqs_file = '../data/alnRes.m8'
human_proteome_file = "../data/human_proteome.fasta"
mapping_file = "../data/processed/protein_list.csv"
# cross validation number
cv_number = 10

# CSV file with 2 columns (PROT A, PROT B)
df = pd.read_csv('../data/processed/HI-union_edges.csv', header=None)

# remove proteins with length < 40 and length > 14000
prot_filter = []
for record in SeqIO.parse(human_proteome_file, "fasta"):
    seq_len = len(record.seq)
    prot_id = record.id.split('|')[1]
    if 40 < seq_len < 14000:
        prot_filter.append(prot_id)

prot_filter = df.isin(prot_filter).all(axis=1)
edges = df[prot_filter]
edges = edges.values

# remove homodimer
edges = remove_self_loop(edges)
nodes = np.unique(edges)

# get prot to index mappings
prot_to_idx = {x:i for i,x in enumerate(nodes)}
idx_to_prot = {i:x for i,x in enumerate(nodes)}
prot_to_idx = np.vectorize(prot_to_idx.get)
idx_to_prot = np.vectorize(idx_to_prot.get)

# create a file for protein mapping
with open(mapping_file, 'w') as f:
    for key in nodes:
        f.write("%s,%s\n"%(key, prot_to_idx(key)))

#undirected and to idx
edges_idx = prot_to_idx(edges)
edges_idx = undirected_edges(edges_idx)

# create multiple sets for cross validation
for cv in range(cv_number):

    # create train, val, test split
    train_idx, val_idx, test_idx = train_test_split(edges_idx, 0.05, 0.45)

    # generate all possible negatives
    negs_idx = negatives(edges_idx)

    # mmseqs file has protein names instead of idx, map back to prot
    # get all proteins similar to the training proteins
    train = idx_to_prot(train_idx)
    train_nodes = np.unique(train)
    train_sim_nodes = similar_proteins(train_nodes, nodes, mmseqs_file, 0.4)
    train_sim_nodes_idx = prot_to_idx(train_sim_nodes)

    # create c1, c2, c3 for negs, val, and test
    c1_negs_idx, c2_negs_idx, c3_negs_idx = c1_c2_c3(train_sim_nodes_idx, negs_idx)
    c1_val_idx, c2_val_idx, c3_val_idx = c1_c2_c3(train_sim_nodes_idx, val_idx)
    c1_test_idx, c2_test_idx, c3_test_idx = c1_c2_c3(train_sim_nodes_idx, test_idx)

    # get the size of each set
    c1_val_len, c2_val_len, c3_val_len = c1_val_idx.shape[0], c2_val_idx.shape[0], c3_val_idx.shape[0]
    c1_test_len, c2_test_len, c3_test_len = c1_test_idx.shape[0], c2_test_idx.shape[0], c3_test_idx.shape[0]

    # create test sets with different pos/neg ratio's
    for x in [1,10,100]:

        np.random.shuffle(c1_negs_idx)
        np.random.shuffle(c2_negs_idx)
        np.random.shuffle(c3_negs_idx)

        # make sure c1, c2, c3 follow the same pos/neg ratio's
        c1_val_negs_idx = c1_negs_idx[:c1_val_len*x]
        c1_test_negs_idx = c1_negs_idx[c1_val_len*x:(c1_val_len*x+c1_test_len*x)]

        c2_val_negs_idx = c2_negs_idx[:c2_val_len*x]
        c2_test_negs_idx = c2_negs_idx[c2_val_len*x:(c2_val_len*x+c2_test_len*x)]

        c3_val_negs_idx = c3_negs_idx[:c3_val_len*x]
        c3_test_negs_idx = c3_negs_idx[c3_val_len*x:(c3_val_len*x+c3_test_len*x)]

        val_negs_idx = np.concatenate([c1_val_negs_idx, c2_val_negs_idx, c3_val_negs_idx])
        test_negs_idx = np.concatenate([c1_test_negs_idx, c2_test_negs_idx, c3_test_negs_idx])

        # create dir for output if not exists
        output_cv_neg = output_path + 'cv/' + str(cv) + '/' + str(x) + '/'
        if not os.path.exists(output_cv_neg):
            os.makedirs(output_cv_neg)

        # save files
        write_set(train_idx, np.zeros((0, 2)), output_cv_neg + 'train.csv')
        write_set(val_idx, val_negs_idx, output_cv_neg + 'val.csv')
        write_set(test_idx, test_negs_idx, output_cv_neg + 'test.csv')
        print(cv, x, 'done')
