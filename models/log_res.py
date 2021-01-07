import pandas as pd
import networkx as nx
import numpy as np
from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import os

def X_y(G, path):
    r"""generate features and labels
    """
    df = pd.read_csv(path)
    y = df['label'].values
    edges = df.iloc[:,:2].values
    X = get_features(G, edges)
    return edges, X, y

def get_features(G, edges):
    r"""get total degree and common common_neighbors
    """
    features = np.zeros((edges.shape[0], 2))
    for i, x in enumerate(edges):
        features[i,1] = G.degree[x[0]]+G.degree[x[1]]
        features[i,0] = len(list(nx.common_neighbors(G, x[0], x[1])))

    return features

# data files
df_proteins = pd.read_csv('../data/processed/protein_list.csv', header=None)
proteins = df_proteins.values[:,1]
processed_path = '../data/processed/'
results_path = '../data/results/'

#for every cross validation
for cv in range(10):
    #get training nodes
    df_train = pd.read_csv(processed_path + 'cv/' + str(cv) + '/1/train.csv', dtype={0:'int', 1:'int'})
    train_edges = df_train.iloc[:,:2].values
    train_nodes = np.unique(train_edges)
    train_to_idx, idx_to_train = get_node_mapping(train_nodes)

    #generate negatives
    train_negatives = negatives(train_to_idx(train_edges), len(train_edges))
    train_negatives = idx_to_train(train_negatives)

    train_edges = np.concatenate([train_edges, train_negatives])

    #create graph
    G = nx.Graph()
    G.add_edges_from(train_edges)
    G.add_nodes_from(proteins)

    #get features and labels
    train_X = get_features(G, train_edges)
    train_y = np.ones(len(train_negatives)*2)
    train_y[len(train_negatives):] = 0

    clf = LogisticRegression(random_state=0)
    clf.fit(train_X, train_y)

    #generate scores for each test set
    for i in [1,10,100]:

        cv_path = processed_path + 'cv/' + str(cv) + '/'+str(i)+'/'
        edges, X_test, y_test = X_y(G, cv_path+'test.csv')
        y_pred = clf.predict_proba(X_test)[:,1]

        df = pd.DataFrame(edges)
        df['label'] = y_test
        df['score'] = y_pred

        output = results_path + 'cv/'+str(cv)+'/'+str(i)+'/DT/'
        if not os.path.exists(output):
            os.makedirs(output)

        df.to_csv(output+'results.csv', index=None)
    print('done', cv)
