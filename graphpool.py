
##pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


##pyg 
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GCNConv,GATConv,dense_mincut_pool,EdgeConv
from torch_geometric import utils


class DeepPoolNet(torch.nn.Module):
    def __init__(self,in_dim,hidden_layers,hidden_dim,n_classes):
        super(DeepPoolNet, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        layers = []
        for i in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_dim,hidden_dim))
        layers.append(nn.Linear(hidden_dim,n_classes))
#         self.linear = nn.Linear(hidden_dim,n_classes)
        self.mlp = nn.Sequential(*layers)
    def forward(self, x, edge_index):
#         print(self.conv1(x, edge_index))
        x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))
        x = F.elu(self.conv4(x, edge_index))
#         x = F.dropout(x)
        s=x
        for layer in self.mlp:
            S = F.elu(layer(s))
#         S = self.linear(x)
#         S = F.dropout(S,p=0.1)
        return x,S



def weighted_density_loss(s,adj,theta=10,EPS=1,debug=False,do_softmax=True):

    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s
    if(do_softmax):
        s = torch.softmax(s, dim=-1)
    ss = torch.matmul(s.transpose(1, 2), s)
    c_size = torch.sum(ss,dim=-1)
    
    if(debug):
        print("c_size:",c_size)
        print(torch.sum(ss,dim=-1))
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    links = torch.einsum('bii->bi', out_adj)
    if(debug):
        print("out_adj:",out_adj)
        print("links:",links)

    density = links/((c_size+EPS)**2)
    
    weights = c_size/(torch.sum(c_size))
    weighted_density = weights*density
    
    density_loss = -5*torch.mean(density)
    
    weighted_density_loss = -theta*torch.mean(weighted_density)

    #normalized weighted_density
    # norm_weighted_density = weighted_density/torch.norm(weighted_density)
    # norm_weighted_density_loss = -theta*torch.mean(norm_weighted_density)

    if(debug):
        print("density:",density)
        print("density norm:",torch.norm(density))
        print("density normed:",density/torch.norm(density))
        print("identity norm:",torch.norm(torch.ones(1,4)))
        print("identity normed:",torch.ones(1,4)/torch.norm(torch.ones(1,4)))
        print("weight:",weights)
        print("weighted_density:",weighted_density)
        print("norm_weighted_density:",weighted_density/torch.norm(weighted_density))

        print("w_loss:",weighted_density_loss)
        
    

    # ortho_loss = torch.norm(density/torch.norm(density) - torch.ones(1,4).cuda()/torch.norm(torch.ones(1,4).cuda()))

    return density_loss, weighted_density_loss 



def main():
    training_epoch = 5000
    N=x.shape[1]
    model = DeepPoolNet(in_dim=N,hidden_layers=1,hidden_dim=32,n_classes=4).cuda()

    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    model.train()


    epoch_losses = []
    t_losses = []
    mincut_losses = []
    d_losses  = []
    o_losses = []
    w_losses = []
    for epoch in range(training_epoch):
        model.zero_grad()
        x_out,s = model(x,edges)
        out, out_adj, mincut_loss, ortho_loss = dense_mincut_pool(x_out,A_norm,s)
        d_loss,w_loss,o_loss = density_loss_P2(s,A,EPS=1)

        loss = w_loss + mincut_loss

        loss.backward()
        optimizer.step()
        epoch_losses.append(loss)
        mincut_losses.append(mincut_loss)