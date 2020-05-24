import random
import os
import matplotlib.pyplot as plt
import networkx as nx
import csv
import numpy as np

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GCNConv,GATConv
import dgl.function as fn
import torch
import torch.nn as nn


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)




# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')
        
def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class PYGNet(torch.nn.Module):
    def __init__(self,in_dim, hidden_dim, n_classes):
        super(PYGNet, self).__init__()
#         self.lin = Sequential(Linear(10, 10))
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, 0, True)
        return self.classify(x)
    
class dglGCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(dglGCNModel, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.in_degrees().view(-1, 1).float().cuda()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

#classGATConv(in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, 
# dropout=0, bias=True, **kwargs)
class pygGATModel(nn.Module):
    def __init__(self,in_dim, hidden_dim, n_classes):
        super(pygGATModel, self).__init__()
#         self.lin = Sequential(Linear(10, 10))
        self.conv1 = GATConv(in_dim, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = torch.mean(x, 0, True)
        return self.classify(x)