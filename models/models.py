import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
import torch_geometric.transforms as T

class GCNModel(torch.nn.Module):
    def __init__(self, num_features):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 128)
        self.conv2 = GCNConv(128, 64)

        self.decoder1 = nn.Linear(64, 16)
        self.decoder2 = nn.Linear(16, 1)

    def forward(self, features, adj, idx):
        dropout = 0.25
        x = F.relu(self.conv1(features, adj))
        x = F.dropout(x, dropout, training=self.training)
        x = F.relu(self.conv2(x, adj))

        x_j = torch.index_select(x, 0, idx[0])
        x_i = torch.index_select(x, 0, idx[1])

        x = torch.abs(x_j - x_i)
        x = torch.sigmoid(self.decoder1(x))
        o = self.decoder2(x)

        return torch.flatten(o)

class LinearModel(torch.nn.Module):
    def __init__(self, num_features):
        super(LinearModel, self).__init__()

        self.conv1 = nn.Linear(num_features, 128)
        self.conv2 = nn.Linear(128, 64)

        self.decoder1 = nn.Linear(64, 16)
        self.decoder2 = nn.Linear(16, 1)

    def forward(self, features, adj, idx):
        dropout = 0.25
        x = F.relu(self.conv1(features))
        x = F.dropout(x, dropout, training=self.training)
        x = F.relu(self.conv2(x))

        x_j = torch.index_select(x, 0, idx[0])
        x_i = torch.index_select(x, 0, idx[1])

        x = torch.abs(x_j - x_i)
        x = torch.sigmoid(self.decoder1(x))
        o = self.decoder2(x)

        return torch.flatten(o)
