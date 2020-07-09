import os.path as osp

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops, structured_negative_sampling)
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from torch_geometric.utils import train_test_split_edges
from datasets.Interactomes import Interactomes
import numpy as np
from utils.explain_outcome import explain_by_node_degree, explain_by_node_degree_pos_neg
from utils.negative_sampling import negatives_sampling_above_distance


torch.manual_seed(12345)

dataset = Interactomes('../../Data/interactomes/human/', 'human')
data = dataset[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0.3, test_ratio=0.2)
# data.x = torch.rand(data.x.shape)

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
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


def get_link_labels(pos_edge_index, neg_edge_index):
    link_labels = torch.zeros(pos_edge_index.size(1) +
                              neg_edge_index.size(1)).float().to(device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels


def train(pos_edge_index, neg_edge_index):
    model.train()
    optimizer.zero_grad()

    link_logits = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

    test_edges = np.concatenate((pos_edge_index.cpu().numpy(),
                                neg_edge_index.cpu().numpy()), axis=1).T
    explain_by_node_degree_pos_neg(pos_edge_index.cpu().numpy().T, test_edges, link_labels.cpu().numpy())

    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()

    return loss


def predict(prefix):
    model.eval()
    pos_edge_index, neg_edge_index = [
        index for _, index in data("{}_pos_edge_index".format(prefix),
                                   "{}_neg_edge_index".format(prefix))
    ]

    link_probs = model(pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    return link_labels, link_probs

### Pos neg samples
x, pos_edge_index = data.x, data.train_pos_edge_index

_edge_index, _ = remove_self_loops(pos_edge_index)
pos_edge_index_with_self_loops, _ = add_self_loops(_edge_index,
                                                   num_nodes=x.size(0))

neg_edge_index = negatives_sampling_above_distance(
    edge_index=pos_edge_index_with_self_loops, train_edges=pos_edge_index_with_self_loops,
    num_nodes=x.size(0), num_neg_samples=pos_edge_index.size(1))

neg_edge_index = torch.stack((neg_edge_index[0], neg_edge_index[1]))

### epochs
for epoch in range(1, 201):

    train_loss = train(pos_edge_index, neg_edge_index)
    link_labels, link_probs = predict('val')
    predict_roc = roc_auc_score(link_labels, link_probs)
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}'
    print(log.format(epoch, train_loss, predict_roc))


# link_labels, link_probs = predict('test')
#
# y_hat = np.round(link_probs)
# test_edges = np.concatenate((data.test_pos_edge_index.detach().cpu().numpy(),
#                             data.test_neg_edge_index.detach().cpu().numpy()), axis=1).T
# print(roc_auc_score(link_labels, link_probs))
# train_edges = data.test_pos_edge_index.detach().cpu().numpy().T
# explain_by_node_degree_pos_neg(train_edges, test_edges, link_labels)
