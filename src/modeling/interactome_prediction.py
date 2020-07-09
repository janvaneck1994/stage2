import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch_geometric.utils import (negative_sampling, structured_negative_sampling,
                                    remove_self_loops, add_self_loops)
from torch_geometric.utils import train_test_split_edges
from datasets.Interactomes import Interactomes
from models.Basic import Basic
import numpy as np
import matplotlib.pyplot as plt
from utils.explain_outcome import explain_by_node_degree

torch.manual_seed(12345)

dataset = Interactomes('../../Data/interactomes/human/', 'human')
data = dataset[0]

# Train/validation/test
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data, val_ratio=0.33, test_ratio=0.33)
data.x = torch.rand(data.x.shape)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Basic(dataset.num_features).to(device), data.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.0001)


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

    neg_edge_index = torch.stack((neg_edge_index[0], neg_edge_index[1]))
    link_logits = model(data, pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)

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

    link_probs = model(data, pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    link_probs = link_probs.detach().cpu().numpy()
    link_labels = link_labels.detach().cpu().numpy()
    predict_roc = roc_auc_score(link_labels, link_probs)
    return link_labels, link_probs

for epoch in range(1, 101):
    train_loss = train()
    link_labels, link_probs = predict('val')
    predict_roc = roc_auc_score(link_labels, link_probs)
    log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}'
    print(log.format(epoch, train_loss, predict_roc))


link_labels, link_probs = predict('test')
y_hat = np.round(link_probs)
test_edges = np.concatenate((data.test_pos_edge_index.detach().cpu().numpy(),
                            data.test_neg_edge_index.detach().cpu().numpy()), axis=1).T

train_edges = data.test_pos_edge_index.detach().cpu().numpy().T
print(data)
explain_by_node_degree(train_edges, test_edges, y_hat, link_labels)

# print(confusion_matrix(link_labels, y_hat))

# plt.hist(link_probs[link_labels == 1], bins=100, alpha=0.5, label='Pos')
# plt.hist(link_probs[link_labels == 0], bins=100, alpha=0.5, label='Neg')
# plt.legend(loc='upper right')
# plt.show()

predict_roc = roc_auc_score(link_labels, link_probs)
print('Test: {:.4f}'.format(predict_roc))
