from sklearn.cluster import SpectralClustering
import numpy as np
import sys
import matplotlib.pyplot as plt
import networkx as nx
import os.path
from os import path
import scipy
import time

class spectarl_clustering:
    def __init__(self,G,threshold=0.01):
        self.G = G
        self.cluster_result = None
        self.graphs_to_process = []
        self.result = {}
        self.threshold = threshold
    def bipartite(self,n_clusters=2):
        adj = nx.adjacency_matrix(self.G)
        round = 0
        self.graphs_to_process.append(self.G)
        nodes = [n for n in self.G.nodes]
        while (self.graphs_to_process):
            round += 1
            print("round ",round)
            for graph in self.graphs_to_process:
                adj = nx.adjacency_matrix(graph)
                cluster_result = SpectralClustering(n_clusters=n_clusters,assign_labels="discretize",random_state=0,affinity='precomputed').fit_predict(adj)
                ## store/update result
                for c in cluster_result:
                    self.result[nodes[c]] = c
                subgraphs = self.extract_subgraphs(cluster_result)
                for sub in subgraphs:
                    ## check if there is only one connected component
                    gc = sub
                    if(nx.number_connected_components(sub) != 1):
                        print("found ",nx.number_connected_components(sub)," connected components")
                        gc = max(nx.connected_components(sub), key=len)

                    L = nx.normalized_laplacian_matrix(gc)
                    e,_ = scipy.sparse.linalg.eigs(L, k=20,which='SM')
                    e.sort()
                    if e[1] < self.threshold:
                        self.graphs_to_process.append(gc)
                self.graphs_to_process.remove(graph)

    def extract_subgraphs(self,cluster_labels):
        subgraph_list = []
        subgraph_label = np.unique(cluster_labels)
        for l in subgraph_label:
            nodes = np.array(self.G.nodes)[np.where(cluster_labels == l)[0]]
            subgraph = self.G.subgraph(nodes)
            subgraph_list.append(subgraph)

        return subgraph_list
    def plot_eigenmap(self,graphs):
        num_graphs = len(graphs)

        fig,axes=plt.subplots(1,num_graphs,figsize=(20,5))
        for idx,graph in enumerate(graphs):
            L = nx.normalized_laplacian_matrix(graph)
            e,_ = scipy.sparse.linalg.eigs(L, k=20,which='SM')
            e.sort()
            x = np.array(range(len(e)))
            axes[idx].set_title("graph:[{}] eigenvalues".format(idx))
            axes[idx].scatter(x[0:20], e[0:20], s=40)

        