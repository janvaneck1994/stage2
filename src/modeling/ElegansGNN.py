import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import train_test_split_edges

from ElegansDataset import ElegansInteractome
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(12345)

dataset = ElegansInteractome('../Data/elegans_ppi_data/')
data = dataset[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0.00, test_ratio=0.75)
# data.x = torch.rand(4924, 1024)

print(data)

nodelist = pd.read_csv('../Data/elegans_ppi_data/raw/elegans_proteins_uniprot_kb.txt', header=None).values.flatten()


human_elegans = pd.read_csv('../Data/elegans_ppi_data/raw/alnRes_human_elegans.m8', sep='\t', header=None)
trained_on = list(set(human_elegans[human_elegans[2] > 0.4][[0,1]].values.flatten()))

pos_test = data.test_pos_edge_index
neg_test = data.test_neg_edge_index

pos_neg = []

for x in range(2):
    pos_neg.append([nodelist[i] for i in pos_test[x]] + [nodelist[i] for i in neg_test[x]])


pos_neg = np.array(pos_neg).T

pos_neg = []

for x in range(2):
    pos_neg.append([i for i in pos_test[x]] + [i for i in neg_test[x]])

pos_neg = np.array(pos_neg).T

train_index = data.train_pos_edge_index.flatten()

idx_c1 = []
idx_c2 = []
idx_c3 = []

for i, x in enumerate(pos_neg):
    if x[0] in train_index and x[1] in train_index:
        idx_c1.append(i)
    elif x[0] not in train_index and x[1] not in train_index:
        idx_c3.append(i)
    else:
        idx_c2.append(i)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        out_weights = 256
        self.conv1 = GCNConv(dataset.num_features, out_weights)
        self.conv2 = GCNConv(dataset.num_features, out_weights)
        self.conv3 = GCNConv(dataset.num_features, out_weights)
        self.conv4 = GCNConv(dataset.num_features, out_weights)

#         self.lin1 = torch.nn.Linear(dataset.num_features, 256)
        self.lin2 = torch.nn.Linear(dataset.num_features*2, 512)

    def forward(self, pos_edge_index, neg_edge_index):

        x1 = F.relu(self.conv1(data.x, data.train_pos_edge_index))
        x2 = F.relu(self.conv2(data.x, data.train_pos_edge_index))
        x3 = F.relu(self.conv3(data.x, data.train_pos_edge_index))
        x4 = F.relu(self.conv4(data.x, data.train_pos_edge_index))

        x = torch.cat([x1,x2,x3,x4, data.x], dim=-1)

        x = F.relu(self.lin2(x))

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])

        einsum = torch.einsum("ef,ef->e", x_i, x_j)
        return einsum



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = torch.load('../models/test_2_100ep.model').to(device), data.to(device)
# model, data = Net().to(device), data.to(device)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)

# model.load_state_dict(torch.load('../models/firsttry.model'))

def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train():
    model.train()
    optimizer.zero_grad()

    x, pos_edge_index = data.x, data.train_pos_edge_index

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=x.size(0),
        num_neg_samples=pos_edge_index.size(1))

    link_logits = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


def test():
    model.eval()
    perfs = []
    for prefix in ["test"]:
        pos_edge_index, neg_edge_index = [
            index for _, index in data("{}_pos_edge_index".format(prefix),
                                       "{}_neg_edge_index".format(prefix))
        ]

        _edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                           num_nodes=data.x.size(0))

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index_with_self_loops, num_nodes=data.x.size(0),
            num_neg_samples=pos_edge_index.size(1)*30)

        link_probs = torch.sigmoid(model(pos_edge_index, neg_edge_index))
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        link_probs = link_probs.detach().cpu().numpy()
        link_labels = link_labels.detach().cpu().numpy()

        # precision, recall, _ = precision_recall_curve(link_labels, link_probs)
        # plt.figure()
        # plt.step(recall, precision, where='post')
        # plt.show()
        perfs.append(roc_auc_score(link_labels[idx_c1], link_probs[idx_c1]))
        perfs.append(roc_auc_score(link_labels[idx_c2], link_probs[idx_c2]))
        perfs.append(roc_auc_score(link_labels[idx_c3], link_probs[idx_c3]))
        print(sum(link_labels[idx_c1]),sum(link_labels[idx_c2]),sum(link_labels[idx_c3]))

        # perfs.append(roc_auc_score(link_labels, link_probs))
    return perfs


best_val_perf = test_perf = 0
for epoch in range(1, 2):
    # train_loss = train()
    test_c1, test_c2, test_c3 = test()
    log = 'Epoch: {:03d},, Test C1: {:.4f}, Test C2: {:.4f}, Test C3: {:.4f}'
    print(log.format(epoch, test_c1, test_c2, test_c3))
