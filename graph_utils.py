from pyparsing import *
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt

import torch
from tqdm import tqdm
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
import os
import numpy as np
import random
from sklearn.manifold import TSNE

from sknetwork.visualization import svg_graph

def build_circuit_graph_undirected(node_list,edge_list):
    g = dgl.DGLGraph()
    g.add_nodes(len(node_list))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)
    
    return g

def build_circuit_graph_directed_sd(node_list,edge_list):
    g = dgl.DGLGraph()
    g.add_nodes(len(node_list))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    return g

def build_circuit_graph_directed_ds(node_list,edge_list):
    g = dgl.DGLGraph()
    g.add_nodes(len(node_list))
    src, dst = tuple(zip(*edge_list))
    g.add_edges(dst, src)
    return g
    

def VisualizeGraph(graph=None,node_labels=None,mode='dgl',ax=None):
    if mode =='dgl':
        graph = graph.to_networkx()

    elif mode =='adj':
        graph = nx.Graph(graph)

    if graph is not None:

        # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
        pos = nx.kamada_kawai_layout(graph)
        plt.figure(1,figsize=(12,12))
        cmap=None
        if node_labels is not None:
            cmap = plt.get_cmap('gist_rainbow',len(np.unique(node_labels)))
            
            cmap.set_under('gray')
        vmin = 0
        vmax = len(np.unique(node_labels)) if node_labels is not None else 37
        nx.draw(graph,pos,with_labels=False,node_size=500,font_size=14,node_color=node_labels,cmap=cmap,vmin = vmin,vmax=vmax,ax=ax)
        # plt.savefig('c432_undirected.png')
        # nx.draw_networkx_labels(nx_G_directed_sd,pos,labels=gate_type_dict2)
        nx.draw_networkx_labels(graph,pos,ax=ax)
        if cmap is not None:
            cm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
            cm._A = []
            plt.colorbar(cm,ax=ax)


def visualize_pyg_gnnexplainer(edge_index, edge_mask, y=None,
                           threshold=None,**kwargs):
        if edge_mask is not None:
            assert edge_mask.size(0) == edge_index.size(1)
        
        if threshold is not None:
            print('Edge Threshold:',threshold)
            edge_mask = (edge_mask >= threshold).to(torch.float)

        subset=[]
        edge_list=[]
        if edge_mask is not None:
            for index,mask in enumerate(edge_mask):
                node_a = edge_index[0,index]
                node_b = edge_index[1,index]
                if node_a not in subset:
                    subset.append(node_a.cpu().item())
                if node_b not in subset:
                    subset.append(node_b.cpu().item())
                if mask:
                    edge_list.append((edge_index[0,index].cpu().item(),edge_index[1,index].cpu().item()))
        else:
            for index,node in edge_index:
                node_a = edge_index[0,index]
                node_b = edge_index[1,index]
                if node_a not in subset:
                    subset.append(node_a.cpu().item())
                if node_b not in subset:
                    subset.append(node_b.cpu().item())
                edge_list.append((edge_index[0,index].cpu().item(),edge_index[1,index].cpu().item()))

        
        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()
        
        data = Data(edge_index=edge_index.cpu(), att=edge_mask, y=y,
                    num_nodes=y.size(0)).to('cpu')

        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
        # mapping = {k: i for k, i in enumerate(subset.tolist())}
        mapping = {k: i for k, i in enumerate(subset)}
        # print(mapping)
        # G = nx.relabel_nodes(G, mapping)

        kwargs['with_labels'] = kwargs.get('with_labels') or True
        kwargs['font_size'] = kwargs.get('font_size') or 10
        kwargs['node_size'] = kwargs.get('node_size') or 200
        kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        # pos = nx.kamada_kawai_layout(G)
        pos = nx.spectral_layout(G)
        ax = plt.figure(figsize=(20,10))

        nx.draw_networkx_nodes(G, pos, **kwargs)
        # print("edge_index:",edge_index.cpu())
        # print("G.edges:",G.edges)
        color = np.array(edge_mask.cpu())
    
        # cmap = plt.get_cmap('gist_rainbow',color)
        # nx.draw(nx_G_undirected,pos,with_labels=False,node_size=500,font_size=14,node_color=node_type,cmap=cmap,vmin = vmin,vmax=vmax)
        
        nx.draw_networkx_edges(G, pos,edgelist=edge_list,
                       width=3, alpha=0.5)
        nx.draw_networkx_labels(G, pos, **kwargs)
        plt.axis('off')
        return plt

def getEdgeList(DGLGraph):
    a = DGLGraph.edges()[0].tolist()
    b = DGLGraph.edges()[1].tolist()
    edges = list(zip(a,b))
    edges = np.array(edges)
    
    return edges

def AdjToEdgelist(adj):
    row = np.where(adj>0)[0]
    col = np.where(adj>0)[1]
    
    edge_list = zip(row,col)
    
    return edge_list

def EdgelistToAdj(edge_list,undirected=True):
    num_nodes = max(edge_list)
    adj = np.zeros((num_nodes,num_nodes))
    for e0,e1 in edge_list:
        adj[e0,e1] = 1
        if(undirected):
            adj[e1,e0] = 1

def ReadEdgeList(path,name):
    prefix = path+name
    filename_adj = prefix + "_A.txt"
    EdgeList = []
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            EdgeList.append((e0,e1))
    return EdgeList

def read_graphfile(datadir, dataname, max_nodes=None, label_edge=False):
    """ Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    logfile = open(datadir+'readgraph_log.txt','w+') ##open a txtfile to save log message
    prefix = os.path.join(datadir, dataname)
    filename_graph_indic = prefix + "_graph_indicator.txt"
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 0
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + "_node_labels.txt"
    node_labels = []
    min_label_val = None
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                l = int(line)
                node_labels += [l]
                if min_label_val is None or min_label_val > l:
                    min_label_val = l
        # assume that node labels are consecutive
        num_unique_node_labels = max(node_labels) - min_label_val + 1
        # node_labels = [l - min_label_val for l in node_labels]
    except IOError:
        print("No node labels")

    filename_node_attrs = prefix + "_node_attributes.txt"
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [
                    float(attr) for attr in re.split("[,\s]+", line) if not attr == ""
                ]
                node_attrs.append(np.array(attrs))
    except IOError:
        print("No node attributes")


    filename_graphs = prefix + "_graph_labels.txt"
    graph_labels = []

    label_vals = []
    try:
        with open(filename_graphs) as f:
            for line in f:
                line = line.strip("\n")
                val = int(line)
                if val not in label_vals:
                    label_vals.append(val)
                graph_labels.append(val)
    except IOError:
        print("No  graph labels")
    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    if label_edge:
        # For Tox21_AHR we want to know edge labels
        filename_edges = prefix + "_edge_labels.txt"
        edge_labels = []

        edge_label_vals = []
        with open(filename_edges) as f:
            for line in f:
                line = line.strip("\n")
                val = int(line)
                if val not in edge_label_vals:
                    edge_label_vals.append(val)
                edge_labels.append(val)

        edge_label_map_to_int = {val: i for i, val in enumerate(edge_label_vals)}

    filename_adj = prefix + "_A.txt"
    adj_list = {i: [] for i in range(0, len(graph_labels))}
    # edge_label_list={i:[] for i in range(1,len(graph_labels)+1)}
    # index_graph = {i: [] for i in range(0, len(graph_labels))}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            logfile.write("Reading edge: ({},{})\n".format(e0,e1))
            if(label_edge):
                edge_label = edge_labels[num_edges]
                adj_list[graph_indic[e0]].append((e0, e1,edge_label))
            else:
                adj_list[graph_indic[e0]].append((e0, e1))
            # index_graph[graph_indic[e0]] += [e0, e1]
            # edge_label_list[graph_indic[e0]].append(edge_labels[num_edges])
            num_edges += 1
    # for k in index_graph.keys():
    #     index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    dgraphs=[]
    for i in range(0, len(adj_list)):
        # indexed from 0 here
        if(label_edge):
            G = nx.Graph()
            G.add_weighted_edges_from(adj_list[i])
            DG=nx.DiGraph()
            DG.add_weighted_edges_from(adj_list[i])
        else:
            G = nx.from_edgelist(adj_list[i])
            DG=nx.DiGraph()
            DG.add_edges_from(adj_list[i])
        # add features and labels
        G.graph["label"] = graph_labels[i]
        DG.graph["label"] = graph_labels[i]
        # Special label for aromaticity experiment
        # aromatic_edge = 2
        # G.graph['aromatic'] = aromatic_edge in edge_label_list[i]

        for u in G.nodes():
            if len(node_labels) > 0:
                # node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u]
                # node_label_one_hot[node_label] = 1
                G.nodes[u]["label"] = node_label
            if len(node_attrs) > 0:
                G.nodes[u]["feat"] = node_attrs[u]

        for u in DG.nodes():
            if len(node_labels) > 0:
                # node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u]
                # node_label_one_hot[node_label] = 1
                DG.nodes[u]["label"] = node_label
            if len(node_attrs) > 0:
                DG.nodes[u]["feat"] = node_attrs[u]
        if len(node_attrs) > 0:
            G.graph["feat_dim"] = node_attrs[0].shape[0]
        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1
        # indexed from 0
        graphs.append(nx.relabel_nodes(G, mapping))
        # dgraphs.append(nx.relabel_nodes(DG, mapping))
        graphs.append(G)
        dgraphs.append(DG)
    return graphs,dgraphs,adj_list

def EmbedClusters(G,cluster_labels,width=1000,height=800,std=10,margin=20):
    nodes = np.array(G.nodes)
    num_cluster = len(np.unique(cluster_labels))
    random.seed(0)

    c_embedding= np.zeros((num_cluster,num_cluster))
    for i in range(num_cluster):
        c_embedding[i,i] = 1
    c_center = TSNE(n_components=2).fit_transform(c_embedding)
    
    # loc_x = np.random.randint(margin,width-margin,num_cluster)
    # loc_y = np.random.randint(margin,height-margin,num_cluster)
    embedding = {}
    for idx,c in enumerate(cluster_labels):
        n = nodes[idx]
        embedding[n]= {}
        embedding[n]['x'] = random.gauss(c_center[c,0],std)
        embedding[n]['y'] = random.gauss(c_center[c,1],std)
    return embedding,c_center

def EmbedClusters_uniform(G,cluster_labels,width=1000,height=800,std=10,margin=20):
    nodes = np.array(G.nodes)
    num_cluster = len(np.unique(cluster_labels))

    x_spacing = width/int(np.sqrt(num_cluster))
    y_spacing = height/int(np.sqrt(num_cluster))
    
    c_center = np.zeros((num_cluster,2))
    for n in range(num_cluster):
        x = n % int(np.sqrt(num_cluster)) * x_spacing
        y = n // int(np.sqrt(num_cluster)) *y_spacing
        c_center[n,0]=x
        c_center[n,1]=y

    # pos = np.zeros((len(cluster_labels),2))
    pos= {}
    for idx,c in enumerate(cluster_labels):
        n = nodes[idx]
        pos[n]={}
        pos[n]['x'] = random.gauss(c_center[c,0],std)
        pos[n]['y'] = random.gauss(c_center[c,1],std)
    return pos,c_center
def DrawGraphToSVGImage(graph,filename,position=None,node_names=None,labels=None,node_size=5,edge_width=0.5,display_edges=True,width=1000,height=800):
    #adjacency – Adjacency matrix of the graph.
    #position – Positions of the nodes.
    #names – Names of the nodes.
    #labels – Labels of the nodes (negative values mean no label).
    position += abs(np.min(position))+1
    adjacency = nx.adjacency_matrix(graph)
    image = svg_graph(adjacency,position=position,names=node_names,labels=labels,node_size=node_size,
                        edge_width=edge_width,display_edges=display_edges,
                        width=width,height = height,filename=filename)


def compress_graph(G):
    node_to_delete=[]
    for u in G.nodes:
        if(not u in node_to_delete):
            neighbors = G.neighbors(u)
            for n in neighbors:
                if((u!=n)&(G.nodes[u]['label']==G.nodes[n]['label'])):
                    G = nx.contracted_nodes(G, u, n)
                    node_to_delete.append(n)
    
    return G

def assignColorbyCluster(nodes,cluster_labels):
    num_labels = len(np.unique(cluster_labels))
    color_step = 255**3//num_labels
    colors = [i*color_step for i in range(num_labels)]
    print(colors)
    node_color = {}
    for index,label in enumerate(cluster_labels):
        n = nodes[index]
        node_color[n] = {}
        node_color[n]['r'] = colors[label]//(255**2)
        node_color[n]['g'] = (colors[label] - node_color[n]['r']*(255**2))//255
        node_color[n]['b'] = colors[label]%(255)
        node_color[n]['a'] = 1
    return node_color
def NXtoGEXF(G,filename,pos,node_size,node_color):
    if pos is None:
        pos = nx.kamada_kawai_layout(G)
    
    for node in np.array(G.nodes):
        G.nodes[node]['viz'] = {}
        G.nodes[node]['viz']['color']={}
        G.nodes[node]['viz']['color']['r'] = node_color[node]['r']
        G.nodes[node]['viz']['color']['g'] = node_color[node]['g']
        G.nodes[node]['viz']['color']['b'] = node_color[node]['b']
        G.nodes[node]['viz']['color']['a'] = node_color[node]['a']
        G.nodes[node]['viz']['position']={}
        G.nodes[node]['viz']['position']['x'] = pos[node]['x']
        G.nodes[node]['viz']['position']['y'] = pos[node]['y']
        G.nodes[node]['viz']['position']['z'] = 0.0

        G.nodes[node]['viz']['size']= node_size

    nx.write_gexf(G, filename,version='1.2draft')

def compute_matrix(nodes,labels,class_size,prediction):
    num_true_class = np.unique(labels)
    num_pred_class = np.unique(prediction)
    print(num_pred_class)
    recall_score = {} 
    precision_score = {}
    f1_score = {}
    cluster_assignment = {}
    for c in num_pred_class:
        cluster_assignment[c] = []
    for label in num_true_class:
        node = nodes[np.where(labels==label)[0]]
        intersect = np.zeros(len(num_pred_class))
        recall = np.zeros(len(num_pred_class))
        precision = np.zeros(len(num_pred_class))
        f1 = np.zeros(len(num_pred_class))
        
        for idx in num_pred_class:
            c = nodes[np.where(prediction==idx)[0]]
            intersect[idx] = len(list(set(node).intersection(c)))

            recall[idx] = intersect[idx] /class_size[label]
            precision[idx] = intersect[idx]/len(c)
            f1[idx] = 2/(1/recall[idx] + 1/precision[idx])
        cluster = np.argmax(intersect)
        if(recall[cluster] > 0.1):
            cluster_assignment[cluster].append(label)
        recall_score[label] = recall[cluster]
        precision_score[label] = precision[cluster]
        f1_score[label] = f1[cluster]
        print("label:",label,"in cluster:",cluster, "Acc:",recall_score[label],"F1:",f1_score[label])

    recall_score_by_cluster = {} 
    precision_score_by_cluster = {}
    f1_score_by_cluster = {}
    for c in cluster_assignment:
        true_labels = cluster_assignment[c]
        true_label_node = []
        size = 0
        for label in true_labels:
            true_label_node += list(nodes[np.where(labels==label)[0]])
            size += class_size[label]
        nodes_in_cluster = nodes[np.where(prediction==c)[0]]

        intersect[idx] = len(list(set(true_label_node).intersection(nodes_in_cluster)))

        
        recall_score_by_cluster[c] = intersect[idx] /size
        precision_score_by_cluster[c] = intersect[idx]/len(nodes_in_cluster)
        f1_score_by_cluster[c] = 2/(1/recall_score_by_cluster[c] + 1/precision_score_by_cluster[c])

    return precision_score,recall_score,f1_score,recall_score_by_cluster,precision_score_by_cluster,f1_score_by_cluster

def compute_confusion_matrix(nodes,labels,prediction):
    num_true_class = np.unique(labels)
    print(num_true_class)
    num_pred_class = np.unique(prediction)
    confusion_matrix = np.zeros((len(num_pred_class),len(num_true_class)))

    for j,label in enumerate(num_true_class):
        node = nodes[np.where(labels==label)[0]]
        for i,idx in enumerate(num_pred_class):
            c = nodes[np.where(prediction==idx)[0]]
            intersect = len(list(set(node).intersection(c)))
            confusion_matrix[i,j] = intersect
    return confusion_matrix

def extract_subgraphs(G,cluster_labels):
    subgraph_list = []
    subgraph_label = np.unique(cluster_labels)
    for l in subgraph_label:
        nodes = np.array(G.nodes)[np.where(cluster_labels == l)[0]]
        subgraph = G.subgraph(nodes)
        subgraph_list.append(subgraph)

    return subgraph_list

def reorder_nodes(G):
    # relabeling
    mapping = {}
    it = 0
    if float(nx.__version__) < 2.0:
        for n in G.nodes():
            mapping[n] = it
            it += 1
    else:
        for n in G.nodes:
            mapping[n] = it
            it += 1
    # indexed from 0
    return nx.relabel_nodes(G, mapping)