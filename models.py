##general
import random
import os
import matplotlib.pyplot as plt
import networkx as nx
import csv
import numpy as np

##dgl
import dgl
import dgl.function as fn

##pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

##pyg 
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GCNConv,GATConv

##customized funcs
from util_funcs import graph_utils

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
        self.GATConv1 = GATConv(in_dim, hidden_dim)
        self.GATConv2 = GATConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.GATConv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
        x = F.relu(self.GATConv2(x, edge_index))
        x = torch.mean(x, 0, True)
        return self.classify(x)

def train_pygmodel(model,dataset,labels,initial_lr=0.001,training_epoch=500,device=None):
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = initial_lr)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    epoch_losses = []
    for epoch in range(training_epoch):
        model.train()
        epoch_loss = 0
        for iter, bg in enumerate(dataset):
    #         prediction=torch.zeros(1,4,dtype=torch.float64).cuda()
            edge_list = graph_utils.getEdgeList(bg)
            
            edges = torch.LongTensor(edge_list.transpose()).to(device)
    #         print(type(edge_list))
    #         edge_list = np.array(edge_list)
            x = bg.in_degrees().view(-1, 1).float().to(device)
            prediction = model(x,edges)
            label = torch.LongTensor(labels[iter]).to(device)
            
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
    #     print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
    return model,epoch_losses,optimizer
def train_dglmodel(model,dataset,labels,initial_lr=0.001,training_epoch=500,device=None):
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = initial_lr)
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    epoch_losses = []
    for epoch in range(training_epoch):
        model.train()
        epoch_loss = 0
        for iter, bg in enumerate(dataset):
    #         prediction=torch.zeros(1,4,dtype=torch.float64).cuda()
            edge_list = graph_utils.getEdgeList(bg)
            
            edges = torch.LongTensor(edge_list.transpose()).to(device)
    #         print(type(edge_list))
    #         edge_list = np.array(edge_list)
            x = bg.in_degrees().view(-1, 1).float().to(device)
            prediction = model(x,edges)
            label = torch.LongTensor(labels[iter]).to(device)
            
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
    #     print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)
    return model,epoch_losses,optimizer

def save_checkpoint(filename,model, optimizer, args, num_epochs=-1, isbest=False, cg_dict=None):
    """Save pytorch model checkpoint.
    
    Args:
        - model         : The PyTorch model to save.
        - optimizer     : The optimizer used to train the model.
        - args          : A dict of meta-data about the model.
        - num_epochs    : Number of training epochs.
        - isbest        : True if the model has the highest accuracy so far.
        - cg_dict       : A dictionary of the sampled computation graphs.
    """
    torch.save(
        {
            "epoch": num_epochs,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cg": cg_dict,
        },
        filename,
    )


def load_ckpt(filename, isbest=False):
    '''Load a pre-trained pytorch model from checkpoint.
    '''
    print("loading model")
    # filename = create_filename(args.ckptdir, args, isbest)
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename)
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")

    # model.load_state_dict(ckpt['model_state_dict'])
    return ckpt
