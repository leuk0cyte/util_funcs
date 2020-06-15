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
import json

def VerilogParser(filename):
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

    return input_port_list,output_port_list,gate_list,module_list
def ReadFlattenVerilogToGraph(filename):
    
    input_port_list,output_port_list,gate_list,module_list = VerilogParser(filename)

    gate_types = {}
    for i in range(len(module_list)):
        #     print(module_list[i][0])
        gate_types[module_list[i][0]] = i+2

    gate_type_dict = {}
    IO_dict = {}

    node_list = []
    edge_list = []

    # Add input ports to gate_dict and gate_type_dict and node_list:
    idx = 0
    for in_port in input_port_list:
        # gate_dict[in_port[0]] = idx
        IO_dict[in_port[0]] = {}
        IO_dict[in_port[0]]['type'] = 0
        IO_dict[in_port[0]]['index'] = idx
        # gate_type_dict[in_port[0]]['inst_name'] = in_port[0]
        IO_dict[in_port[0]]['type_name'] = 'Input'
        node_list.append(idx)
        idx += 1

    # Add gates to gate_type_dict and node_list:
    for a_gate in gate_list:
        # if is not a submodule
        if(a_gate[0][0] in module_list):
            node_list.append(idx)
            if("DFFSR" in a_gate[0][0]):
                # special handle for D-Fflip-flopSR
                gate_name = a_gate[-3][1]
            else:
                gate_name = a_gate[-1][1]
            gate_type_dict[gate_name] = {}
            gate_type_dict[gate_name]['index'] =idx
            gate_type_dict[gate_name]['type'] = gate_types[a_gate[0][0]]
            # gate_type_dict[gate_name]['inst'] = a_gate[0][1]
            gate_type_dict[gate_name]['type_name'] = a_gate[0][0]
            # gate_type_dict[gate_name]['gate_name'] = a_gate[-1][1]
            idx += 1


    # Add output ports to node_list and gate_dict:
    for out_port in output_port_list:

        IO_dict[out_port[0]]={}
        IO_dict[out_port[0]]['index'] = idx
        IO_dict[out_port[0]]['type'] = 1
        IO_dict[out_port[0]]['type_name'] = 'Output'
        node_list.append(idx)

        idx += 1
    # Append vcc and gnd to the end of gate_type_dict and node_list
    IO_dict["1'b0"] = {}
    IO_dict["1'b0"]['index']=idx
    node_list.append(idx)
    IO_dict["1'b0"]['type'] = 0
    IO_dict["1'b0"]['type_name'] = 'Gnd'
    idx += 1
    IO_dict["1'b1"] = {}
    IO_dict["1'b1"]['index']=idx
    node_list.append(idx)
    IO_dict["1'b1"]['type'] = 0
    IO_dict["1'b1"]['name'] = 'Vcc'
    idx += 1

    # Add connections to edge_list:
    for idx, out_port in enumerate(output_port_list):
        # node_list.append(idx+existing_length)
        # node_list.append(gate_type_dict[out_port[0]+'_port']['index'])
        edge_list.append((gate_type_dict[out_port[0]]['index'], IO_dict[out_port[0]]['index'])) ##connection btw buffer and output port

    # Add connections to edge_list:
    for a_gate in gate_list:
        # add all connections including a self-loop
        for connection in a_gate[1:]:
            if(connection[1] in IO_dict):
                ##if connect to input port
                edge_list.append(
                    (IO_dict[connection[1]]['index'], gate_type_dict[a_gate[-1][1]]['index']))
            elif("DFFSR" in a_gate[0][0]):
                # special handle for DFFSR;
                edge_list.append(
                    (gate_type_dict[connection[1]]['index'], gate_type_dict[a_gate[-3][1]]['index']))
            else:
                edge_list.append(
                    (gate_type_dict[connection[1]]['index'], gate_type_dict[a_gate[-1][1]]['index']))
                
        # gate_type_dict[gate_dict[a_gate[-1][1]]]['connection'] = edges  

    module_dict={}
    module_dict['IO'] = {}
    module_dict['Gates'] = {}
    module_dict['IO'] = IO_dict
    module_dict['Gates'] = gate_type_dict
    # existing_length = len(node_list)

    # build dgl graph from component list
    # G_directed_sd = graph_utils.build_circuit_graph_directed_sd(node_list,edge_list)
    # G_undirected = graph_utils.build_circuit_graph_undirected(node_list,edge_list)
    Gx = nx.Graph()
    Gx.add_nodes_from(node_list)
    Gx.add_edges_from(edge_list)
    # check isolated nodes and remove if exists
    isolated_nodes = list(nx.isolates(Gx))
    Gx.remove_nodes_from(isolated_nodes)
    G = dgl.DGLGraph()
    G.from_networkx(Gx)

    return G,module_dict

def ReadHierarchicalVerilogToGraph(modulepath, topmodule_name,current_node_idx,IsTopModule,logfile=None):
    

    edge_list = []
    node_list = []
    
    filename = modulepath + "/qflow/synthesis/" + topmodule_name + '.rtlbb.v'

    module_dict={}


    input_port_list,output_port_list,gate_list,module_list = VerilogParser(filename)
    modules = [module_list[n][0] for n in range(len(module_list))]
    
    gate_types = {}
    for i in range(len(module_list)):
        #     print(module_list[i][0])
        gate_types[module_list[i][0]] = i+2
    gate_type_dict = {}
    IO_dict = {}

    ##node index adjustment
    idx = current_node_idx
    # Append vcc and gnd to the end of gate_type_dict and node_list
    if(IsTopModule):
        logfile = open(modulepath+'/log.txt','w+') ##open a txtfile to save log message
        #only create pwer and ground in top module
        node_list.append(idx)
        idx += 1
        logfile.write("---Adding Gnd at index:{}---\n".format(idx))
        node_list.append(idx)
        idx += 1
        logfile.write("---Adding Vcc at index:{}---\n".format(idx))
    PowerAndGnd = {}
    PowerAndGnd["1'b0"]={}
    PowerAndGnd["1'b0"]['index']=0
    PowerAndGnd["1'b0"]['type'] = 0
    PowerAndGnd["1'b0"]['type_name'] = 'Gnd'
    PowerAndGnd["1'b0"]['connections'] = {}
    PowerAndGnd["1'b1"]={}
    PowerAndGnd["1'b1"]['index']=1
    PowerAndGnd["1'b1"]['type'] = 0
    PowerAndGnd["1'b1"]['type_name'] = 'Vcc'
    PowerAndGnd["1'b1"]['connections'] = {}
    # Add input ports to gate_dict and gate_type_dict and node_list:
    logfile.write("---Adding IO for:{}---\n".format(topmodule_name))
    for in_port in input_port_list:
        # gate_dict[in_port[0]] = idx
        IO_dict[in_port[0]] = {}
        IO_dict[in_port[0]]['type'] = 0
        IO_dict[in_port[0]]['index'] = idx
        IO_dict[in_port[0]]['type_name'] = 'Input'
        IO_dict[in_port[0]]['connections'] = {}
        node_list.append(idx)
        logfile.write("{}: Added {} at index:{}\n".format(topmodule_name,in_port[0],idx))
        idx += 1
    # Add output ports to IO_dict and nodelist:
    for out_port in output_port_list:

        IO_dict[out_port[0]]={}
        IO_dict[out_port[0]]['index'] = idx
        IO_dict[out_port[0]]['type'] = 1
        IO_dict[out_port[0]]['type_name'] = 'Output'
        IO_dict[out_port[0]]['connections'] = {}
        node_list.append(idx)
        logfile.write("{}: Added {} at index:{}\n".format(topmodule_name,out_port[0],idx))
        idx += 1
    logfile.write("---Finished reading IO for:{}---\n".format(topmodule_name))
    # Add gates to gate_type_dict and node_list:
    module_dict['Modules'] = {}
    logfile.write("---Adding gates and submodules for:{}---\n".format(topmodule_name))
    for a_gate in gate_list:
        if(a_gate[0][0] in modules):
            # if is not a submodule
            node_list.append(idx)
            
            if("DFFSR" in a_gate[0][0]):
                # special handle for D-Fflip-flopSR
                gate_name = a_gate[-3][1]
            else:
                gate_name = a_gate[-1][1]
            gate_type_dict[gate_name] = {}
            gate_type_dict[gate_name]['index'] =idx
            gate_type_dict[gate_name]['type'] = gate_types[a_gate[0][0]]
            # gate_type_dict[gate_name]['inst'] = a_gate[0][1]
            gate_type_dict[gate_name]['type_name'] = a_gate[0][0]
            gate_type_dict[gate_name]['gate_name'] = a_gate[0][1]
            gate_type_dict[gate_name]['connections'] = {}
            # gate_type_dict[gate_name]['gate_name'] = a_gate[-1][1]
            logfile.write("Added {} at index:{}\n".format(gate_name,idx))
            idx += 1
        else:
            #if it is a submodule
            #recursively call
            print("{}: Found submodules: {} in {}\n".format(topmodule_name,a_gate[0][1],topmodule_name))
            logfile.write("{}: Found submodules: {} in {}\n".format(topmodule_name,a_gate[0][1],topmodule_name))
            module_dict['Modules'][a_gate[0][1]]={}
            _,module_inst,new_node_idx,module_node_list,module_edge_list = ReadHierarchicalVerilogToGraph(modulepath, a_gate[0][0],current_node_idx=idx,IsTopModule=False,logfile=logfile)
            node_list += module_node_list
            edge_list += module_edge_list
            idx=new_node_idx
            module_dict['Modules'][a_gate[0][1]] = module_inst

            for connection in a_gate[1:]:
                if(connection[0] in module_inst['IO']):
                    logfile.write("{}: Mapping connection: {}\n".format(topmodule_name,connection))
                    if(module_inst['IO'][connection[0]]['type_name'] == "Output"):
                        #check if the gate is already in gate_dict
                        if((not connection[1] in gate_type_dict) and (not connection[1] in IO_dict) and (not connection[1] in PowerAndGnd)):
                            logfile.write("{}: Creating Output node: {} at idx:{}\n".format(topmodule_name,connection[1],idx))
                            gate_type_dict[connection[1]]={}
                            gate_type_dict[connection[1]]['index'] = idx 
                            gate_type_dict[connection[1]]['type'] =  len(module_list) + 2
                            gate_type_dict[connection[1]]['type_name'] = 'IO_Interface'
                            gate_type_dict[connection[1]]['connections'] = {}
                            node_list.append(idx)
                            idx+=1
                        if(connection[1] in IO_dict):
                            output_node = module_inst['IO'][connection[0]]['index']
                            logfile.write("{}: Adding connection: {} to {}\n".format(topmodule_name,output_node,IO_dict[connection[1]]['index']))
                            edge_list.append((output_node,IO_dict[connection[1]]['index']))
                            IO_dict[connection[1]]['connections'][a_gate[0][0]+':'+connection[0]]=output_node
                        elif(connection[1] in PowerAndGnd):
                            output_node = module_inst['IO'][connection[0]]['index']
                            logfile.write("{}: Adding connection: {} to {} at index:{}\n".format(topmodule_name,output_node,PowerAndGnd[connection[1]]['type_name'],PowerAndGnd[connection[1]]['index']))
                            edge_list.append((output_node,PowerAndGnd[connection[1]]['index']))
                        elif(connection[1] in gate_type_dict):
                            output_node = module_inst['IO'][connection[0]]['index']
                            logfile.write("{}: Adding connection: {} to {}\n".format(topmodule_name,output_node,gate_type_dict[connection[1]]['index']))
                            edge_list.append((output_node,gate_type_dict[connection[1]]['index']))
                            gate_type_dict[connection[1]]['connections'][a_gate[0][0]+':'+connection[0]]=output_node
                        ##add connection to edgelist
                        
                    elif(module_inst['IO'][connection[0]]['type_name'] == "Input"):
                        ## if connect to components input.
                        if ((not connection[1] in gate_type_dict) and (not connection[1] in IO_dict) and (not connection[1] in PowerAndGnd)):
                            logfile.write("{}: Creating Input node: {} at idx:{}\n".format(topmodule_name,connection[1],idx))
                            gate_type_dict[connection[1]]={}
                            gate_type_dict[connection[1]]['index'] = idx 
                            gate_type_dict[connection[1]]['type'] =  len(module_list) + 2
                            gate_type_dict[connection[1]]['type_name'] = 'IO_Interface'
                            gate_type_dict[connection[1]]['connections'] = {}
                            node_list.append(idx)
                            idx+=1
                        if(connection[1] in IO_dict):
                            input_node = module_inst['IO'][connection[0]]['index']
                            logfile.write("{}: Adding connection: {} to {}\n".format(topmodule_name,IO_dict[connection[1]]['index'],input_node))
                            edge_list.append((IO_dict[connection[1]]['index'],input_node))
                            IO_dict[connection[1]]['connections'][a_gate[0][0]+':'+connection[0]]=input_node
                        elif(connection[1] in PowerAndGnd):
                            input_node = module_inst['IO'][connection[0]]['index']
                            logfile.write("{}: Adding connection: {}:at index {} to {} \n".format(topmodule_name,PowerAndGnd[connection[1]]['type_name'],PowerAndGnd[connection[1]]['index'],input_node))
                            edge_list.append((PowerAndGnd[connection[1]]['index'],input_node))
                        elif(connection[1] in gate_type_dict):
                            input_node = module_inst['IO'][connection[0]]['index']
                            logfile.write("{}: Adding connection: {} to {}\n".format(topmodule_name,gate_type_dict[connection[1]]['index'],input_node))
                            edge_list.append((gate_type_dict[connection[1]]['index'],input_node))
                            gate_type_dict[connection[1]]['connections'][a_gate[0][0]+':'+connection[0]]=input_node

    logfile.write("---Finished Adding gates for:{}---\n".format(topmodule_name))
    print("---Finished Adding gates for:{}---\n".format(topmodule_name))
    # Add connections to edge_list:
    for out_port in output_port_list:
        edge_list.append((gate_type_dict[out_port[0]]['index'], IO_dict[out_port[0]]['index'])) ##connection btw buffer and output port
        IO_dict[out_port[0]]['connections']={}
        IO_dict[out_port[0]]['connections'][out_port[0]]=gate_type_dict[out_port[0]]['index']
    # Add connections to edge_list:
    for a_gate in gate_list:
        # add all connections including a self-loop
        if(a_gate[0][0] in modules):
            if("DFFSR" in a_gate[0][0]):
                gate_name = a_gate[-3][1]
            else:
                gate_name = a_gate[-1][1]

            gate_type_dict[gate_name]['connections']={}
            for connection in a_gate[1:]:
                if(connection[1] in IO_dict):
                    ##if connect to input port
                    edge_list.append(
                        (IO_dict[connection[1]]['index'], gate_type_dict[gate_name]['index']))
                    gate_type_dict[gate_name]['connections'][connection[1]]=IO_dict[connection[1]]['index']
                elif(connection[1] in PowerAndGnd):
                    ##if connect to Vcc or Gnd
                    edge_list.append(
                        (PowerAndGnd[connection[1]]['index'], gate_type_dict[gate_name]['index']))
                    gate_type_dict[gate_name]['connections'][connection[1]]=PowerAndGnd[connection[1]]['index']
                else:
                    edge_list.append(
                        (gate_type_dict[connection[1]]['index'], gate_type_dict[gate_name]['index']))
                    gate_type_dict[gate_name]['connections'][connection[1]]=gate_type_dict[connection[1]]['index']



    module_dict['IO'] = {}
    module_dict['Gates'] = {}
    module_dict['IO'] = IO_dict
    module_dict['Gates'] = gate_type_dict
    module_dict['PowerAndGnd'] = {}
    module_dict['PowerAndGnd'] = PowerAndGnd

    Gx = nx.Graph()
    Gx.add_nodes_from(node_list)
    Gx.add_edges_from(edge_list)
    # check isolated nodes and remove if exists
    isolated_nodes = list(nx.isolates(Gx))
    Gx.remove_nodes_from(isolated_nodes)
    G = dgl.DGLGraph()
    G.from_networkx(Gx)
    # output module dict to json

    savepath = modulepath+'/graphs/'
    if(IsTopModule):
        logfile.write("------Generating graph data files------\n")
        if(not path.exists(savepath)):
            os.mkdir(savepath)
            print('Created Dir: ',savepath)
        with open(savepath+'module_dict.json', 'w') as fp:
            json.dump(module_dict, fp)
        print('Saved module dict to: ',savepath+'module_dict.json')

        np.savetxt(savepath+topmodule_name+'_nodelist.txt',node_list,"%s",delimiter=",")
        np.savetxt(savepath+topmodule_name+'_A.txt',edge_list,"%s",delimiter=",")

        node_labels = np.zeros((len(node_list),1))
        readme = []
        node_labels,readme,current_idx = GetNodeLabelFromModuleDict(module_dict,readme,node_labels,starting_idx=0,logfile=logfile)
        idx = current_idx
        node_idx = [module_dict['Gates'][g]['index'] for g in module_dict['Gates']]
        node_idx += [module_dict['IO'][g]['index'] for g in module_dict['IO']]
        node_labels[node_idx]=idx
        readme.append((topmodule_name,idx))

        graph_labels=[]
        graph_labels.append(0)
        graph_indicator = np.zeros((len(node_list),1))
        np.savetxt(savepath+topmodule_name+'_node_labels.txt',node_labels,"%d",delimiter=",")
        np.savetxt(savepath+topmodule_name+'_graph_labels.txt',graph_labels,"%d",delimiter=",")
        np.savetxt(savepath+topmodule_name+'_graph_indicator.txt',graph_indicator,"%d",delimiter=",")
        np.savetxt(savepath+topmodule_name+'_ReadMe.txt',readme,"%s",delimiter=",")

        print('Done...Saved logfile to: ',savepath+'log.txt')

    else:
        logfile.write("---Finished processing {}---\n".format(topmodule_name))
        print("---Finished processing {}---\n".format(topmodule_name))
        logfile.write("---Returning...---\n")
        print("---Returning...---\n")
    return G,module_dict,idx,node_list,edge_list
def GetNodeLabelFromModuleDict(module_dict,readme,node_labels,starting_idx=0,logfile=None):
    current_idx = starting_idx
    idx=starting_idx
    for mod in module_dict['Modules']:
        logfile.write("starting idx:{};  idx adjust to:{}\n".format(starting_idx,idx))
        if(bool(module_dict['Modules'][mod]['Modules'])):
            logfile.write("found submodules in: {};  starting_idx is:{}\n".format(mod,idx))
            node_labels,readme,current_idx = GetNodeLabelFromModuleDict(module_dict['Modules'][mod],readme,node_labels,starting_idx=idx,logfile=logfile)
            idx=current_idx
        logfile.write("current_idx :{};  idx adjust to:{}\n".format(current_idx,idx))
        node_idx = [module_dict['Modules'][mod]['Gates'][g]['index'] for g in module_dict['Modules'][mod]['Gates']]
        node_idx += [module_dict['Modules'][mod]['IO'][g]['index'] for g in module_dict['Modules'][mod]['IO']]
        node_labels[node_idx]=idx
        readme.append((mod,idx))
        idx+=1
        current_idx=idx
    return node_labels,readme,current_idx

def search(module,mod_name,index):
    for gate in module['Gates']:
        a_gate = module['Gates'][gate]
        if(a_gate['index'] == index):
            print(mod_name,":Gate:",gate)
    for gate in module['IO']:
        a_gate = module['IO'][gate]
        if(a_gate['index'] == index):
            print(mod_name,":IO:",gate)

def search_module(module_dict,index):
    for mod in module_dict['Modules']:
        mod_dict = module_dict['Modules'][mod]
        search(mod_dict,mod,index)
def search_index(module_dict,index):
    ##search top level
    search(module_dict,"top",index)
    #search each module
    if(bool(module_dict['Modules'])):
        search_module(module_dict,index)
# INCOMPLETE
def ReadGraphAndVerify(graphsPath,verilogPath,modulename):
    graph_prefix = graphsPath + '/' + modulename
    filename_adj = graph_prefix + "_A.txt"
    EdgeList = []
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            EdgeList.append((e0,e1))
    #create a networkx graph
    G = nx.Graph()
    G.add_edges_from(EdgeList)
    #check number of connected components
    n = nx.number_connected_components(G)
    print("Number of connected components in graph:",n)

    #reading original verilogs
    verilog_prefix = verilogPath + '/' + verilogPath
    filename_topmodule = verilog_prefix + modulename + ".rtlbb.v"
    input_port_list,output_port_list,gate_list,module_list = VerilogParser(filename_topmodule)
    modules = [module_list[n][0] for n in range(len(module_list))]
    # for a_gate in gate_list:
    #     if(a_gate[0][0] in modules):
    #     else:
    #         module_filename = verilog_prefix + a_gate[0][0] + ".rtlbb.v"
    #         input_port_list,output_port_list,gate_list,module_list = VerilogParser(module_filename)

    print("Verifying edges ...")


