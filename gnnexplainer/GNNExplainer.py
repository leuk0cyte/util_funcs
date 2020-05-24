from math import sqrt

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
import numpy as np
EPS = 1e-15


class GNNExplainer(torch.nn.Module):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.

    .. note::

        For an example of using GNN-Explainer, see `examples/gnn_explainer.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        gnn_explainer.py>`_.

    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
    """

    coeffs = {
        'edge_size': 0.005,
        'node_feat_size': 1.0,
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, epochs=100, lr=0.01, log=True,writer=None):
        super(GNNExplainer, self).__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.writer = writer

    def __set_masks__(self, x, edge_index, init="normal"):
        (N, F), E = x.size(), edge_index.size(1)

        std = 0.1
        self.node_feat_mask = torch.nn.Parameter(torch.randn(F) * 0.1)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)
#         self.edge_mask = torch.nn.Parameter(torch.zeros(E)*50)
        
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask
        
    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_masks = None
        self.edge_mask = None

    def __num_hops__(self):
        num_hops = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                num_hops += 1
        return num_hops

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __subgraph__(self, node_idx, x, edge_index, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)

        if node_idx is not None:
            subset, edge_index, edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=num_nodes, flow=self.__flow__())
            x = x[subset]
        else:
            x=x
            edge_index=edge_index
            row, col = edge_index
            edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
            edge_mask[:]=True

        
        for key, item in kwargs:
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, edge_mask, kwargs

    def __loss__(self, node_idx, log_logits, pred_label):
        loss = -log_logits[node_idx, pred_label[node_idx]]
        m = self.edge_mask.sigmoid()
        loss = loss + self.coeffs['edge_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        m = self.node_feat_mask.sigmoid()
        loss = loss + self.coeffs['node_feat_size'] * m.sum()
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss
    def __graph_loss__(self, log_logits, pred_label,epoch):
        log_logits.retain_grad()
        pred_loss = -torch.log(log_logits[0][pred_label])
#         pred_loss = -torch.log(log_logits[0,pred_label])
        pred_loss.retain_grad()
        m = self.edge_mask.sigmoid()
        m.retain_grad()
        size_loss = self.coeffs['edge_size'] * m.sum()
        
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m+ EPS)
        ent.retain_grad()
        mask_ent_loss = self.coeffs['edge_ent'] * ent.mean()
        
        loss = pred_loss + size_loss + mask_ent_loss
#         loss = pred_loss
#         loss.retain_grad()
#         loss.backward()
#         pred_loss.backward()
#         print("first backward")
#         print("edge_mask:",self.model.conv2.__edge_mask__.grad)
#         print("log_logits",log_logits.grad)
#         print("Second backward")
#         size_loss.backward(retain_graph=True)
#         print("edge_mask:",self.model.conv2.__edge_mask__.grad)
        
#         mask_ent_loss.backward()
#         print("Third backward")
#         print("mask_ent_loss",mask_ent_loss.grad)
#         print("log_logits",log_logits.grad)
#         print("m",m.grad)
#         print("edge_mask:",self.model.conv2.__edge_mask__.grad)

#         loss = pred_loss
#         m = self.node_feat_mask.sigmoid()
#         loss = loss + self.coeffs['node_feat_size'] * m.sum()
#         ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
#         loss = loss + self.coeffs['node_feat_ent'] * ent.mean()
        
        if self.writer is not None:
            self.writer.add_scalar("optimization/size_loss", size_loss, epoch)
#             self.writer.add_scalar("optimization/feat_size_loss", feat_size_loss, epoch)
            self.writer.add_scalar("optimization/mask_ent_loss", mask_ent_loss, epoch)
            # self.writer.add_scalar('optimization/grad_loss', grad_loss, epoch)
            self.writer.add_scalar("optimization/pred_loss", pred_loss, epoch)
#             self.writer.add_scalar("optimization/lap_loss", lap_loss, epoch)
            self.writer.add_scalar("optimization/overall_loss", loss, epoch)

        return loss
    def explain_node(self, node_idx, x, edge_index, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, hard_edge_mask, kwargs = self.__subgraph__(
            node_idx, x, edge_index, **kwargs)

        # Get the initial prediction.
        with torch.no_grad():
            log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)

        optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask],
                                     lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description(f'Explain node {node_idx}')
        epoch_losses=[]
        for epoch in range(1, self.epochs + 1):
            epoch_loss=0
            optimizer.zero_grad()
            h = x * self.node_feat_mask.view(1, -1).sigmoid()
            log_logits = self.model(x=h, edge_index=edge_index, **kwargs)
            loss = self.__loss__(0, log_logits, pred_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            epoch_losses.append(epoch_loss)
            if self.log:  # pragma: no cover
                pbar.update(1)
        
        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return node_feat_mask, edge_mask,epoch_losses


    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None,**kwargs):
        r"""Visualizes the subgraph around :attr:`node_idx` given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. (default: :obj:`None`)
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.pyplot`
        """

        assert edge_mask.size(0) == edge_index.size(1)
        
        if threshold is not None:
            print('Edge Threshold:',threshold)
            edge_mask = (edge_mask >= threshold).to(torch.float)
          
        if node_idx is not None:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, hard_edge_mask = k_hop_subgraph(
                node_idx, self.__num_hops__(), edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

            edge_mask = edge_mask[hard_edge_mask]
        else:
            subset=[]
            for index,mask in enumerate(edge_mask):
                node_a = edge_index[0,index]
                node_b = edge_index[1,index]
                if node_a not in subset:
                    subset.append(node_a.cpu().item())
#                     print("add: "+node_a)
                if node_b not in subset:
                    subset.append(node_b.cpu().item())
#                     print("add: "+node_b)
#             subset = torch.cat(subset).unique()
        edge_list=[]
        for index, edge in enumerate(edge_mask):
            if edge:
                edge_list.append((edge_index[0,index].cpu(),edge_index[1,index].cpu()))
        
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        data = Data(edge_index=edge_index.cpu(), att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')

        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
#         mapping = {k: i for k, i in enumerate(subset.tolist())}
        mapping = {k: i for k, i in enumerate(subset)}
#         print(mapping)
#         G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 200
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        pos = nx.spring_layout(G)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="-",
                    alpha=max(data['att'], 0.1),
                    shrinkA=sqrt(kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(kwargs['node_size']) / 2.0,
#                     connectionstyle="arc3,rad=0.1",
                ))
# #         if node_feature_mask is not None:
        nx.draw_networkx_nodes(G, pos, **kwargs)

        color = np.array(edge_mask.cpu())

        nx.draw_networkx_edges(G, pos,
                       width=3, alpha=0.5, edge_color=color,edge_cmap=plt.cm.Reds)
        nx.draw_networkx_labels(G, pos, **kwargs)
        plt.axis('off')
        return plt
    
    def mask_density(self):
        mask_sum = torch.sum(self.edge_mask).cpu()
        adj_sum = len(self.edge_mask)
        return mask_sum / adj_sum
    
    def explain_graph(self, x, edge_index,device=None, **kwargs):
  
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(device)
        self.model.eval()
        self.__clear_masks__()

        num_edges = edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        x, edge_index, hard_edge_mask, kwargs = self.__subgraph__(node_idx=None,x=x, edge_index=edge_index, **kwargs)
        # Get the initial prediction.
        with torch.no_grad():
            x = x.to(device)
            edge_index = edge_index.to(device)
            log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
            probs_Y = torch.softmax(log_logits, 1)
            pred_label = probs_Y.argmax(dim=-1)
            print("Predicted label:",pred_label)
        self.__set_masks__(x, edge_index)
        self.to(x.device)
    
        ## optimizer imported to original GNN repo
#         params = [self.edge_mask]
#         filter_fn = filter(lambda p : p.requires_grad, params)
#         optimizer = torch.optim.Adam(filter_fn, lr=0.01)
        ##end
        
        optimizer = torch.optim.Adam([self.edge_mask],lr=self.lr)
        print("Learning rate:",self.lr)

        epoch_losses=[]
        for epoch in range(1, self.epochs + 1):
            epoch_loss=0
            optimizer.zero_grad()
            # h = x * self.node_feat_mask.view(1, -1).sigmoid()

            log_logits = self.model(x=x, edge_index=edge_index, **kwargs)
#             print("log_logits:",log_logits)
            pred = torch.softmax(log_logits, 1)
            loss = self.__graph_loss__(pred, pred_label,epoch)
            loss.backward()

            optimizer.step()
#             print("mask grad:,",self.edge_mask.grad)
            epoch_loss += loss.detach().item()
#             print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
            epoch_losses.append(epoch_loss)
            print("epoch:",epoch,"; loss:",loss,"; pred:", pred)

        node_feat_mask = self.node_feat_mask.detach().sigmoid()
        edge_mask = self.edge_mask.new_zeros(num_edges)
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()

        return edge_mask,node_feat_mask,epoch_losses
        
    def __repr__(self):
        return f'{self.__class__.__name__}()'