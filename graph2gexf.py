import graph_utils
import networkx as nx
import numpy as np

module_name = 'fir_18bCo_18bIn'
graph_dir = f'./{module_name}'

graphs, _, _ = graph_utils.read_graphfile(graph_dir, module_name, max_nodes=None, label_edge=True)

G = graphs[1]

# node_color = [{}]
# node_color.append()

graph_utils.NXtoGEXF(G,f'./{module_name}/{module_name}.gexf',None,None,None)