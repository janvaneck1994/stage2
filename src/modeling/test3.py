import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import train_test_split_edges
from datasets.Interactomes import Interactomes
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(12345)

dataset = Interactomes('../../Data/interactomes/human/', 'human')
data = dataset[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0, test_ratio=0.5)
data.x = torch.rand(data.x.shape)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128)
        self.conv2 = GCNConv(128, 64)

    def forward(self, pos_edge_index, neg_edge_index):

        x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
        x = self.conv2(x, data.train_pos_edge_index)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])
        return torch.einsum("ef,ef->e", x_i, x_j)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def total_neighbors(G, x, y):
    return len(list(G.neighbors(x))) + len(list(G.neighbors(y)))

def train(pos_edge_index, neg_edge_index):
    model.train()
    optimizer.zero_grad()


    link_logits = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


def predict(pos_edge_index, neg_edge_index):
    model.eval()

    # pos_edge_index =
    #
    # _edge_index, _ = remove_self_loops(pos_edge_index)
    # pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
    #                                                    num_nodes=x.size(0))
    #
    # neg_edge_index = negative_sampling(
    #     edge_index=pos_edge_index_with_self_loops, num_nodes=len(list(set(pos_edge_index.cpu().numpy().flatten()))),
    #     num_neg_samples=pos_edge_index.size(1)*60)

    link_probs = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    return link_labels, link_probs


def get_negatives(x, pos_edge_index, mult):

    _edge_index, _ = remove_self_loops(pos_edge_index)
    pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                       num_nodes=x.size(0))

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index_with_self_loops, num_nodes=len(list(set(pos_edge_index.cpu().numpy().flatten()))),
        num_neg_samples=pos_edge_index.size(1)*mult)

    map_dict = {x:i for i,x in enumerate(set(pos_edge_index.cpu().numpy().flatten()))}

    f = np.vectorize(lambda x: map_dict[x])
    pos_edge = f(pos_edge_index.cpu().numpy())

    G = nx.Graph()
    G.add_nodes_from(pos_edge.flatten())
    G.add_edges_from(pos_edge.T)

    pos = np.array([total_neighbors(G, x[0], x[1]) for x in pos_edge.T])
    neg = np.array([total_neighbors(G, x[0], x[1]) for x in neg_edge_index.cpu().numpy().T])

    shortest_path = []
    for x in neg_edge_index.cpu().numpy().T:
        try:
            shortest_path.append(len(nx.shortest_path(G, x[0], x[1])))
        except:
            shortest_path.append(999)

    neg = neg[np.array(shortest_path) < 4]

    neg_edge_index = neg_edge_index[:,np.array(shortest_path) < 4]

    print(len(neg))
    print(len(pos))

    all_cn = [pos, neg]
    max_cn = max(map(lambda x: max(x), all_cn))
    min_cn = min(map(lambda x: min(x), all_cn))
    print(len(list(nx.isolates(G))))

    labels = ['pos', 'neg']
    plt.figure(figsize=(8,6))
    plt.hist(all_cn, bins=100, histtype='bar', label=labels)
    plt.legend(loc="upper right")
    plt.title('Neg/Pos with combined common neighbors (SkipGNN paper)')
    plt.xlabel('Combined number of neighbors of A and B')
    plt.ylabel('Frequency')
    plt.show()

    return neg_edge_index


x, train_pos_edge_index = data.x, data.train_pos_edge_index

test_pos_edge_index = data.test_pos_edge_index
train_negatives = get_negatives(x, train_pos_edge_index, 1)

test_sample = torch.cat([train_pos_edge_index, test_pos_edge_index, train_negatives], axis=1)
test_negatives = get_negatives(x, test_sample, 1)

print(train_pos_edge_index.size(), test_pos_edge_index.size())
for epoch in range(1, 501):
    x, pos_edge_index = data.x, data.train_pos_edge_index
    train_loss = train(train_pos_edge_index, train_negatives)
    link_labels, link_probs = predict(test_pos_edge_index, test_negatives)
    predict_roc = roc_auc_score(link_labels, link_probs)
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}'
    print(log.format(epoch, train_loss, predict_roc))
