from pyparsing import *
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt

import torch
from tqdm import tqdm
import os
import numpy as np
import os.path
from os import path

def ReadFlattenVerilogToGraph(filename):
    with open(filename, 'rt') as fh:
        code = fh.read()

    # add support for ground input '1'b0'
    gnd = Literal("1'b0")
    vcc = Literal("1'b1")
    identifier = Word(alphas+"_", alphanums+"_"+'['+']') | gnd | vcc

    input_port = Suppress("input") + delimitedList(identifier) + Suppress(";")
    output_port = Suppress("output") + \
        delimitedList(identifier) + Suppress(";")
    # wire_list = Suppress("wire") + delimitedList(identifier) + Suppress(";")
    # gate_type = oneOf("not and or nand nor xor")
    port = Group(Suppress('.') + identifier +
                 Suppress("(") + identifier + Suppress(")"))
    gate = Group(identifier + identifier) + \
        Suppress("(") + delimitedList(port) + Suppress(");")

    # module_title = Suppress("module") + identifier + Suppress("(") + delimitedList(identifier) + Suppress(");")
    # module = module_title + input_port + output_port + Optional(wire_list) + OneOrMore(gate) + Suppress("endmodule")

    input_port_list = input_port.searchString(code)
    output_port_list = output_port.searchString(code)
    gate_list = gate.searchString(code)

    with open('/usr/local/share/qflow/tech/osu035/osu035_stdcells.v', 'rt') as fh:
        code_std_cells = fh.read()

    module_title = Suppress("module") + identifier + \
        Suppress("(") + delimitedList(identifier) + Suppress(");")
    module_list = module_title.searchString(code_std_cells)

    gate_types = {}
    for i in range(len(module_list)):
        #     print(module_list[i][0])
        gate_types[module_list[i][0]] = i+2
    # Building graph components
    gate_dict = {}  # gate name mapped to unique integers
    gate_type_dict = {}
    # define gate types
    #input_port = 0
    #output_port = 1
    # other gates start from 2
    node_list = []
    edge_list = []

    # Add input ports to gate_dict and gate_type_dict and node_list:
    idx = 0
    for in_port in input_port_list:
        gate_dict[in_port[0]] = idx
        gate_type_dict[idx] = 0
        node_list.append(idx)
        idx += 1

    # Add gates to gate_dict and node_list:
    for a_gate in gate_list:
        node_list.append(idx)
        if("DFFSR" in a_gate[0][0]):
            # special handle for D-Fflip-flopSR
            gate_dict[a_gate[-3][1]] = idx
            gate_type_dict[idx] = gate_types[a_gate[0][0]]
        else:
            gate_dict[a_gate[-1][1]] = idx
            gate_type_dict[idx] = gate_types[a_gate[0][0]]
        idx += 1

    # Add output ports to node_list and gate_dict:
    for out_port in output_port_list:
        gate_type_dict[idx] = 1  # gate type is output port
        node_list.append(idx)
        gate_dict[out_port[0]] = idx
        idx += 1
    # Append vcc and gnd to the end of gate_dict and node_list
    gate_dict["1'b0"] = idx
    node_list.append(idx)
    gate_type_dict[idx] = 0
    idx += 1
    gate_dict["1'b1"] = idx
    node_list.append(idx)
    gate_type_dict[idx] = 0
    idx += 1

    existing_length = idx
    # Add connections to edge_list:
    for idx, out_port in enumerate(output_port_list):
        node_list.append(idx+existing_length)
        edge_list.append((gate_dict[out_port[0]], idx+existing_length))

    # Add connections to edge_list:
    for a_gate in gate_list:
        #     print(a_gate)
        # add all connections including a self-loop
        for connection in a_gate[1:]:
            if("DFFSR" in a_gate[0][0]):
                # special handle for DFFSR;
                edge_list.append(
                    (gate_dict[connection[1]], gate_dict[a_gate[-3][1]]))
            else:
                edge_list.append(
                    (gate_dict[connection[1]], gate_dict[a_gate[-1][1]]))

    existing_length = len(node_list)

    # build dgl graph from component list
    # G_directed_sd = graph_utils.build_circuit_graph_directed_sd(node_list,edge_list)
    # G_undirected = graph_utils.build_circuit_graph_undirected(node_list,edge_list)
    Gx = nx.Graph()
    Gx.add_nodes_from(node_list)
    Gx.add_edges_from(edge_list)
    # check isolated nodes and remove if exists
    Gx.remove_nodes_from(list(nx.isolates(Gx)))
    G = dgl.DGLGraph()
    G.from_networkx(Gx)

    return G, gate_dict, input_port_list, output_port_list


def ReadHierarchicalVerilogToGraph(modulepath, topmodule_name):
    filepath = modulepath + '/qflow/synthesis/'
    gate_label = {}
    label = 0
    edge_list = []
    node_list = []
    
    IO_type_dict = {}
    # componentlist=[]
    module_dict={}
    module_num=0
    for filename in os.listdir(filepath):
        if(filename.endswith("rtlbb.v")):
            if(filename != (topmodule_name+'.rtlbb.v')):
                IO_dict = {}

                print(filename)
                modulename = filename.split('.')[0]
                # componentlist.append(module)
                module_num+=1
                module_inst = {}
                module_inst['name'] = modulename
                module_inst['Index'] = module_num

                existing_length = len(gate_label)
                G, gate_dict, input_port_list, output_port_list = ReadFlattenVerilogToGraph(
                    filepath+'/'+filename)

                a = G.edges()[0].tolist()
                b = G.edges()[1].tolist()
                # node index adjustment
                a = (i + existing_length for i in a)
                b = (i + existing_length for i in b)
                edges = list(zip(a, b))

                nodes = G.nodes()
                # node index adjustment
                nodes = list(n + existing_length for n in nodes)
                # num_merge = 0 ##keep track of the number of port merged due to same IO
                
                for i in input_port_list:
                    IO_inst = {}
                    IO_inst['name'] = i[0]
                    IO_inst['index'] = gate_dict[i[0]]+existing_length
                    IO_inst['type'] = 'Input'
                    IO_dict[i[0]] = IO_inst
                    ## If input overlap
                    for m in module_dict:
                        module_iter = module_dict[m]
                        for p in module_iter['IO']:
                            Port = module_iter['IO'][p]
                            if((Port['type'] == 'Input') & (Port['name']==IO_inst['name'])):
                                print("Found duplicate Input:",Port['name']," in modlue:",module_iter['name'],"index,",Port['index'])
                                print("IO_inst['index']",IO_inst['index'])
                                IO_inst['index'] = Port['index']
                                print("Port['index']",Port['index'])
                                for idx, (e0, e1) in enumerate(edges):
                                    ##redirect edges
                                    if(e0 == gate_dict[i[0]]+existing_length):
                                        print("merged:", e0, " To:", IO_inst['index'])
                                        edges[idx] = (IO_inst['index'], e1)
                                    elif(e1 == gate_dict[i[0]]+existing_length):
                                        print("merged:", e1, " To:", IO_inst['index'])
                                        edges[idx] = (e0, IO_inst['index'])
                
                for o in output_port_list:
                    # IO_dict[gate_dict[o[0]]+existing_length] = o[0]
                    IO_inst = {}
                    IO_inst['name'] = o[0]
                    IO_inst['index'] = gate_dict[o[0]]+existing_length
                    IO_inst['type'] = 'Output'
                    IO_dict[o[0]] = IO_inst

                for i in range(len(G.nodes())):
                    gate_label[existing_length+i] = label
                label += 1

                module_inst['IO'] = IO_dict
                module_dict[module_inst['name']] = module_inst

                edge_list = edge_list+edges
                node_list = node_list+nodes
                print(len(node_list))
            else:
                print("skipping ", filename)
                
    ## read in top module
    filename = filepath+topmodule_name+'.rtlbb.v'
    print("Reading:",filename)
    with open(filename, 'rt') as fh:
        code = fh.read()

    # add support for ground input '1'b0'
    gnd = Literal("1'b0")
    vcc = Literal("1'b1")
    # add support for ground input '1'b0'
    identifier = Word(alphas+"_", alphanums+"_"+'['+']') | gnd |vcc
    input_port = Suppress("input") + delimitedList(identifier) + Suppress(";")
    output_port = Suppress("output") + delimitedList(identifier) + Suppress(";")
    # wire_list = Suppress("wire") + delimitedList(identifier) + Suppress(";")
    # gate_type = oneOf("not and or nand nor xor")
    port = Group(Suppress('.') + identifier + Suppress("(") + identifier + Suppress(")"))
    gate = Group(identifier + identifier) + Suppress("(") + delimitedList(port) + Suppress(");")
    input_port_list = input_port.searchString(code)
    output_port_list = output_port.searchString(code)
    gate_string = gate.searchString(code)
    with open('/usr/local/share/qflow/tech/osu035/osu035_stdcells.v', 'rt') as fh:
        code_std_cells = fh.read()

    module_title = Suppress("module") + identifier + Suppress("(") + delimitedList(identifier) + Suppress(");")
    module_list = module_title.searchString(code_std_cells)

    gate_types={}
    for i in range(len(module_list)):
    #     print(module_list[i][0])
        gate_types[module_list[i][0]] = i+2


    ## Building graph components
    gate_dict = {} #gate name mapped to unique integers
    gate_type_dict = {}
    mips_node_list = []
    mips_edge_list = []

    existing_length = len(node_list)
    #Add gnd pin at the beginning:
    gate_dict["1'b0"] = 0 + existing_length
    gate_type_dict[0+existing_length] = 0 
    gate_dict["1'b1"] = 1 + existing_length
    gate_type_dict[1+existing_length] = 0
    gate_list=[]

    gate_length = 0
    for a_gate in gate_string:
        if(a_gate[0][0] in module_dict):
            module = module_dict[a_gate[0][0]]
            for connection in a_gate[1:]:
                ## if come from outputs of components, add gate to gate_dict and add connection to components outputs(search from IO_dict)
                if(connection[0] in module['IO']):
                    if(module['IO'][connection[0]]['type'] == "Output"):
                        #check if the gate is already in gate_dict
                        if(not connection[1] in gate_dict):
                            ## add gate connecting to outputs of components to gate_dict 
                            gate_dict[connection[1]] = gate_length + 2 + existing_length
                            print("creating new gate:",connection[1],": ",gate_length + 2 + existing_length)
                            gate_length +=1
                        ##add connection to edgelist
                        output_node = module['IO'][connection[0]]['index']
                        print("Adding connection:",output_node," to ",gate_dict[connection[1]])
                        edge_list.append((output_node,gate_dict[connection[1]]))
                    
                    ## if connect to components input.
                    elif(module['IO'][connection[0]]['type'] == "Input"):
                        if (not connection[1] in gate_dict):
                            ## add gate connecting to outputs of components to gate_dict 
                            gate_dict[connection[1]] = gate_length + 2 + existing_length
                            print("creating new gate:",connection[1],": ",gate_length + 2 + existing_length)
                            gate_length +=1
                        #connect gate to inputs of components
                        input_node = module['IO'][connection[0]]['index']
                        print("Adding connection:",gate_dict[connection[1]]," to ",input_node)
                        edge_list.append((gate_dict[connection[1]],input_node))
                    
                    
        else:
            gate_list.append(a_gate)
            
    existing_length = existing_length + gate_length + 2
    #Add input ports to gate_dict and gate_type_dict:
    for idx,in_port in enumerate(input_port_list):
        if(not in_port[0] in gate_dict):
            gate_dict[in_port[0]] = idx+existing_length
            gate_type_dict[idx+existing_length] = 0
            existing_length+=1
    #Add gates to gate_dict: 
    for idx, a_gate in enumerate(gate_list):
        if("DFFSR" in a_gate[0][0]):
            #special handle for D-Fflip-flopSR
            gate_dict[a_gate[-3][1]] = idx  + existing_length
            gate_type_dict[idx +existing_length] = gate_types[a_gate[0][0]]
        else:
            gate_dict[a_gate[-1][1]] = idx  + existing_length
            gate_type_dict[idx +existing_length] = gate_types[a_gate[0][0]]
    
    current_node_length = len(node_list)

    #Add input ports and gates to node_list:
    for idx in range(len(gate_dict)):
        mips_node_list.append(idx+current_node_length) 

    #Add connections to edge_list:
    for a_gate in gate_list:
        ##add all connections including a self-loop
        for connection in a_gate[1:]:
            if("DFFSR" in a_gate[0][0]):
                #special handle for DFFSR;
                mips_edge_list.append((gate_dict[connection[1]],gate_dict[a_gate[-3][1]]))
            else:
                mips_edge_list.append((gate_dict[connection[1]],gate_dict[a_gate[-1][1]]))
                
    # existing_length = len(node_list)
    current_node_length = len(mips_node_list) +current_node_length

    #Add output ports to node_list:
    for idx,out_port in enumerate(output_port_list):
        gate_type_dict[idx+current_node_length] = 1 #gate type is output port
        mips_node_list.append(idx+current_node_length)
        mips_edge_list.append((gate_dict[out_port[0]],idx+current_node_length))                    
    node_list = node_list + mips_node_list
    edge_list = edge_list + mips_edge_list


    savepath = modulepath +'/graphs/'

    if(not path.exists(savepath)):
        os.mkdir(savepath)
        print('Created Dir: ',savepath)

    np.savetxt(savepath+topmodule_name+'_nodelist.txt',node_list,"%s",delimiter=",")
    np.savetxt(savepath+topmodule_name+'_A.txt',edge_list,"%s",delimiter=",")

    for node in mips_node_list:
        gate_label[node] = len(module_dict)
    node_label = []
    graph_indicator=[]

    for i in range(len(gate_label)):
        node_label.append((gate_label[i]))
        graph_indicator.append(0)

    graph_labels = []
    graph_labels.append(0)
    np.savetxt(savepath+topmodule_name+'_graph_indicator.txt',graph_indicator,"%s",delimiter=",")
    np.savetxt(savepath+topmodule_name+'_graph_labels.txt',graph_labels,"%s",delimiter=",")
    np.savetxt(savepath+topmodule_name+'_node_labels.txt',node_label,"%s",delimiter=",")
    com_lib=[]

    for idx,c in enumerate(module_dict):
        com_lib.append((idx,c))
    com_lib.append((len(module_dict),topmodule_name))
    np.savetxt(savepath+topmodule_name+'_Readme.txt',com_lib,"%s",delimiter="    ")

    return IO_dict