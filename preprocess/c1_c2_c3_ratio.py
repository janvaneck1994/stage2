import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
import seaborn as sns
from utils import train_test_split,\
                    remove_self_loop,\
                    undirected_edges,\
                    similar_proteins,\
                    c1_c2_c3

output_path = '../data/processed/'
mmseqs_file = '../data/alnRes.m8'
human_proteome_file = "../data/human_proteome.fasta"
mapping_file = "../data/processed/protein_list.csv"

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

#undirected and to idx
edges_idx = prot_to_idx(edges)
edges_idx = undirected_edges(edges_idx)

train_perc_list = []
train_list = []
C1_list = []
C2_list = []
C3_list = []
C3_list_abs = []

# create sets for different train ratios
for i in range(0,10):
    test_perc = i/10
    train_perc = 1-test_perc

    train_idx, val_idx, test_idx = train_test_split(edges_idx, val_ratio=0, test_ratio=test_perc)
    train = idx_to_prot(train_idx)
    train_nodes = np.unique(train)
    train_sim_nodes = similar_proteins(train_nodes, nodes, mmseqs_file, 0.4)
    train_sim_nodes_idx = prot_to_idx(train_sim_nodes)
    c1_test_idx, c2_test_idx, c3_test_idx = c1_c2_c3(train_sim_nodes_idx, test_idx)

    total_edges = len(edges_idx)/2
    train_list.append((len(train_idx)/2)/total_edges)
    train_perc_list.append(train_perc)
    C1_list.append(len(c1_test_idx)/total_edges)
    C2_list.append(len(c2_test_idx)/total_edges)
    C3_list.append(len(c3_test_idx)/total_edges)
    C3_list_abs.append(len(c3_test_idx))

# plot the size of each difficulty set for every train ratio
sns.set()
plt.plot(train_perc_list, train_list, marker='o', color='red', label="train")
plt.plot(train_perc_list, C1_list, marker='o', color='blue', label="C1")
plt.plot(train_perc_list, C2_list, marker='o', color='green', label="C2")
plt.plot(train_perc_list, C3_list, marker='o', color='purple', label="C3")
plt.xlabel('Train ratio')
plt.ylabel('Dataset ratio')
plt.title('C1, C2, C3 distribution for different train/test ratio\'s')
plt.legend()
plt.show()

# zoom in on the C# plot
plt.plot(train_perc_list, C3_list_abs, marker='o', color='purple', label="C3")
plt.xlabel('Train ratio')
plt.ylabel('Frequency')
plt.title('C3 frequency for different train/test ratio\'s')
plt.legend()
plt.show()
