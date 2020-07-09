import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv, SAGEConv, SGConv, GATConv, GINConv  # noqa
from torch.nn import Sequential, Linear, ReLU


class Basic(torch.nn.Module):
    def __init__(self, num_features):
        super(Basic, self).__init__()
        self.conv1 = GCNConv(num_features, 256, improved=True)
        self.conv2 = GCNConv(256, 128,  improved=True)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, data, pos_edge_index, neg_edge_index):

        x = F.relu(self.conv1(data.x, data.train_pos_edge_index))
        x = self.conv2(x, data.train_pos_edge_index)

        total_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        x_j = torch.index_select(x, 0, total_edge_index[0])
        x_i = torch.index_select(x, 0, total_edge_index[1])

        x = torch.add(x_i,x_j)
        x = F.relu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        return torch.flatten(x)
        # return torch.einsum("ef,ef->e", x_i, x_j)
