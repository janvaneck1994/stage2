import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_tpfptnfn(edges, y_hat, link_labels):

    tp_edges = edges[np.all([y_hat==1, link_labels==1], axis=0)]
    fp_edges = edges[np.all([y_hat==1, link_labels==0], axis=0)]
    tn_edges = edges[np.all([y_hat==0, link_labels==0], axis=0)]
    fn_edges = edges[np.all([y_hat==0, link_labels==1], axis=0)]

    return np.array([tp_edges, fp_edges, tn_edges, fn_edges])

def get_posneg(edges, link_labels):

    pos = edges[link_labels==1]
    neg = edges[link_labels==0]

    return np.array([pos, neg])

def total_common_neighbors(G, x, y):
    return len(list(nx.common_neighbors(G, x, y)))

def total_neighbors(G, x, y):
    return len(list(G.neighbors(x))) + len(list(G.neighbors(y)))

def explain_by_node_degree(train_edges, test_edges, y_hat, link_labels):

    tp, fp, tn, fn = get_tpfptnfn(test_edges, y_hat, link_labels)

    nodes = set(np.concatenate((train_edges, test_edges)).flatten())

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(train_edges)

    tp_cn = [total_neighbors(G, x[0], x[1]) for x in tp]
    fp_cn = [total_neighbors(G, x[0], x[1]) for x in fp]
    tn_cn = [total_neighbors(G, x[0], x[1]) for x in tn]
    fn_cn = [total_neighbors(G, x[0], x[1]) for x in fn]
    all_cn = [tp_cn, fn_cn, fp_cn, tn_cn]

    max_cn = max(map(lambda x: max(x), all_cn))
    min_cn = min(map(lambda x: min(x), all_cn))

    labels = ['tp', 'fn', 'fp', 'tn']
    plt.hist(all_cn, bins=np.arange(max_cn), histtype='bar', stacked=True, label=labels)
    plt.legend(loc="upper right")
    plt.title('Correctness of classification')
    plt.xlabel('Combined number of neighbors of A and B')
    plt.ylabel('Frequency')
    plt.show()

def explain_by_node_degree_pos_neg(train_edges, test_edges, link_labels):

    pos, neg = get_posneg(test_edges, link_labels)

    nodes = set(np.concatenate((train_edges, test_edges)).flatten())

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(train_edges)

    pos_cn = [total_neighbors(G, x[0], x[1]) for x in pos]
    neg_cn = [total_neighbors(G, x[0], x[1]) for x in neg]
    all_cn = [pos_cn, neg_cn]

    max_cn = max(map(lambda x: max(x), all_cn))
    min_cn = min(map(lambda x: min(x), all_cn))

    labels = ['pos', 'neg']
    plt.hist(all_cn, bins=np.arange(max_cn), histtype='bar', stacked=True, label=labels)
    plt.legend(loc="upper right")
    plt.title('Correctness of classification')
    plt.xlabel('Combined number of neighbors of A and B')
    plt.ylabel('Frequency')
    plt.show()
