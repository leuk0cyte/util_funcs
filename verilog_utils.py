from pyparsing import *
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt

import torch
from tqdm import tqdm
import os,fnmatch
import numpy as np
import os.path
from os import path
import json
from datetime import datetime

class verilog_reader():
    def __init__(
        self,
        top_module_name,
        filepath,
    ):
        self.filepath = filepath
        self.top_module_name = top_module_name
        self.wire_dict = {}
        self.wire_dict["1'b0"] = {}
        self.wire_dict["1'b1"] = {}
        self.wire_dict["1'b0"]['source']=0
        self.wire_dict["1'b1"]['source']=1
        self.wire_dict["1'b0"]['destination'] = []
        self.wire_dict["1'b1"]['destination'] = []
        self.module_dict = {}
        self.PowerAndGnd = {}
        self.PowerAndGnd["1'b0"]={}
        self.PowerAndGnd["1'b0"]['index']=0
        self.PowerAndGnd["1'b0"]['type'] = 0
        self.PowerAndGnd["1'b0"]['type_name'] = 'Gnd'
        self.PowerAndGnd["1'b0"]['connections'] = {}
        self.PowerAndGnd["1'b1"]={}
        self.PowerAndGnd["1'b1"]['index']=1
        self.PowerAndGnd["1'b1"]['type'] = 0
        self.PowerAndGnd["1'b1"]['type_name'] = 'Vcc'
        self.PowerAndGnd["1'b1"]['connections'] = {}
        # self.current_node_idx = 0

    def VerilogParser(self,filename):
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
        gnd = Literal("1'b0")
        vcc = Literal("1'b1")
        with open('/usr/local/share/qflow/tech/osu035/osu035_stdcells.v', 'rt') as fh:
            code_std_cells = fh.read()

        module_title =  identifier + Suppress(";")
        module_list = module_title.searchString(code_std_cells)

        return input_port_list,output_port_list,gate_list,module_list
    


    def ReadHierarchicalVerilogToGraph(self,topmodule_name,current_node_idx,IsTopModule,logfile=None):
        
        modulepath = self.filepath
        # current_node_idx = self.current_node_idx

        edge_list = []
        node_list = []
        
        filename = modulepath + "/qflow/synthesis/" + topmodule_name + '.rtlbb.v'

        module_dict = {}
        gate_type_dict = {}
        IO_dict = {}
        gate_types = {}

        input_port_list,output_port_list,gate_list,module_list = self.VerilogParser(filename)
        modules = [module_list[n][0] for n in range(len(module_list))]
        

        for i in range(len(module_list)):
            #     print(module_list[i][0])
            gate_types[module_list[i][0]] = i+2


        ##node index adjustment
        idx = current_node_idx
        # Append vcc and gnd to the end of gate_type_dict and node_list
        if(IsTopModule):
            now = datetime.now()
            logfile = open(modulepath+'/log_'+str(now)+'.txt','w+') ##open a txtfile to save log message
            #only create pwer and ground in top module
            node_list.append(idx)
            idx += 1
            logfile.write("---Adding Gnd at index:{}---\n".format(idx))
            node_list.append(idx)
            idx += 1
            logfile.write("---Adding Vcc at index:{}---\n".format(idx))

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

            self.wire_dict[topmodule_name+in_port[0]] = {}
            self.wire_dict[topmodule_name+in_port[0]]['source'] = IO_dict[in_port[0]]['index']
            self.wire_dict[topmodule_name+in_port[0]]['destination'] = []
            logfile.write("{}: Creating wire: {}, source:{}, destination:[]\n".format(topmodule_name,topmodule_name+in_port[0],IO_dict[in_port[0]]['index']))

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
            self.wire_dict[topmodule_name+out_port[0]] = {}
            self.wire_dict[topmodule_name+out_port[0]]['source'] = []
            self.wire_dict[topmodule_name+out_port[0]]['destination'] = []
            self.wire_dict[topmodule_name+out_port[0]]['destination'].append(IO_dict[out_port[0]]['index'])
            logfile.write("{}: Creating wire: {}, source:[], destination:{}\n".format(topmodule_name,topmodule_name+out_port[0],IO_dict[out_port[0]]['index']))

            idx += 1
        logfile.write("---Finished reading IO for:{}---\n".format(topmodule_name))
        # Add gates to gate_type_dict and node_list:
        module_dict['Modules'] = {}
        logfile.write("---Adding gates and submodules for:{}---\n".format(topmodule_name))
        for a_gate in gate_list:
            if(a_gate[0][0] in modules):
                # if is not a submodule
                node_list.append(idx)
                gate_name = a_gate[1][0]
                if("DFFSR" in a_gate[0][0]):
                    # special handle for D-Fflip-flopSR
                    wire_name = a_gate[-3][1]
                    
                else:
                    wire_name = a_gate[-1][1]
                gate_type_dict[wire_name] = {}
                gate_type_dict[wire_name]['index'] =idx
                gate_type_dict[wire_name]['type'] = gate_types[a_gate[0][0]]
                # gate_type_dict[gate_name]['inst'] = a_gate[0][1]
                gate_type_dict[wire_name]['type_name'] = a_gate[0][0]
                gate_type_dict[wire_name]['gate_name'] = a_gate[0][1]
                gate_type_dict[wire_name]['connections'] = {}
                # gate_type_dict[gate_name]['gate_name'] = a_gate[-1][1]
                logfile.write("Added {} at index:{}\n".format(wire_name,idx))

                self.wire_dict[topmodule_name+wire_name] = {}
                self.wire_dict[topmodule_name+wire_name]['source'] = gate_type_dict[wire_name]['index']
                self.wire_dict[topmodule_name+wire_name]['destination'] = []
                logfile.write("{}: Creating wire: {}, source:{}, destination:[]\n".format(topmodule_name,topmodule_name+wire_name,gate_type_dict[wire_name]['index']))
                idx += 1
            else:
                #if it is a submodule
                #recursively call
                print("{}: Found submodules: {} in {}\n".format(topmodule_name,a_gate[0][1],topmodule_name))
                logfile.write("{}: Found submodules: {} in {}\n".format(topmodule_name,a_gate[0][1],topmodule_name))
                module_dict['Modules'][a_gate[0][1]]={}
                _,module_inst,new_node_idx,module_node_list,module_edge_list = self.ReadHierarchicalVerilogToGraph(a_gate[0][0],current_node_idx=idx,IsTopModule=False,logfile=logfile)
                node_list += module_node_list
                edge_list += module_edge_list
                # self.wire_dict.update(module_wire_dict)
                idx=new_node_idx
                module_dict['Modules'][a_gate[0][1]] = module_inst

                for connection in a_gate[1:]:
                    if(connection[0] in module_inst['IO']):
                        logfile.write("{}: Mapping connection: {}\n".format(topmodule_name,connection))
                        if(module_inst['IO'][connection[0]]['type_name'] == "Output"):
                            #check if the gate is already in gate_dict
                            if(not topmodule_name+connection[1] in self.wire_dict):
                                self.wire_dict[topmodule_name+connection[1]] = {}
                                self.wire_dict[topmodule_name+connection[1]]['source'] = []
                                self.wire_dict[topmodule_name+connection[1]]['destination'] = []
                                logfile.write("{}: Creating wire: {}, source:{}, destination:[]\n".format(topmodule_name,topmodule_name+connection[1],module_inst['IO'][connection[0]]['index']))
                                
                            self.wire_dict[topmodule_name+connection[1]]['source'] = module_inst['IO'][connection[0]]['index']
                                
                                # node_list.append(idx)
                        elif(module_inst['IO'][connection[0]]['type_name'] == "Input"):
                            ## if connect to components input
                            if(connection[1] in self.PowerAndGnd):
                                ##if connect to Vcc or Gnd
                                logfile.write("{}: Adding connection to power/ground: source:{}, destination:{}\n".format(topmodule_name,connection[1],module_inst['IO'][connection[0]]['index']))
                                self.wire_dict[connection[1]]['destination'].append(module_inst['IO'][connection[0]]['index'])
                                logfile.write("test: {}, {}\n".format(connection[1],self.wire_dict[connection[1]]['destination']))
                                # edge_list.append(
                                #     (self.PowerAndGnd[connection[1]]['index'], module_inst['IO'][connection[0]]['index']))
                            else:
                                if(not topmodule_name+connection[1] in self.wire_dict):
                                    self.wire_dict[topmodule_name+connection[1]] = {}
                                    self.wire_dict[topmodule_name+connection[1]]['source'] = []
                                    self.wire_dict[topmodule_name+connection[1]]['destination'] = []
                                    logfile.write("{}: Creating wire: {}, source:[], destination:{}\n".format(topmodule_name,topmodule_name+connection[1],module_inst['IO'][connection[0]]['index']))
                                self.wire_dict[topmodule_name+connection[1]]['destination'].append(module_inst['IO'][connection[0]]['index'])
                            

                                # idx+=1

        logfile.write("---Finished Adding gates for:{}---\n".format(topmodule_name))
        print("---Finished Adding gates for:{}---\n".format(topmodule_name))
        # Add connections to edge_list:
        for out_port in output_port_list:
            edge_list.append((gate_type_dict[out_port[0]]['index'], IO_dict[out_port[0]]['index'])) ##connection btw buffer and output port
            IO_dict[out_port[0]]['connections']={}
            IO_dict[out_port[0]]['connections'][out_port[0]] = gate_type_dict[out_port[0]]['index']
            self.wire_dict[topmodule_name+out_port[0]]['source'] = gate_type_dict[out_port[0]]['index']
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
                        self.wire_dict[topmodule_name+connection[1]]['destination'].append(gate_type_dict[gate_name]['index'])

                        edge_list.append(
                            (IO_dict[connection[1]]['index'], gate_type_dict[gate_name]['index']))
                        gate_type_dict[gate_name]['connections'][connection[1]]=IO_dict[connection[1]]['index']
                    elif(connection[1] in self.PowerAndGnd):
                        ##if connect to Vcc or Gnd
                        logfile.write("{}: Adding connection to power/ground: source:{}, destination:{}\n".format(topmodule_name,connection[1],gate_type_dict[gate_name]['index']))
                        self.wire_dict[connection[1]]['destination'].append(gate_type_dict[gate_name]['index'])
                        logfile.write("test: {}, {}\n".format(connection[1],self.wire_dict[connection[1]]['destination']))
                        edge_list.append(
                            (self.PowerAndGnd[connection[1]]['index'], gate_type_dict[gate_name]['index']))
                        gate_type_dict[gate_name]['connections'][connection[1]] = self.PowerAndGnd[connection[1]]['index']
                    
                    elif(connection[1] in gate_type_dict):
                        self.wire_dict[topmodule_name+connection[1]]['destination'].append(gate_type_dict[gate_name]['index'])
                        edge_list.append(
                            (self.wire_dict[topmodule_name+connection[1]]['source'], gate_type_dict[gate_name]['index']))
                        gate_type_dict[gate_name]['connections'][connection[1]] = gate_type_dict[connection[1]]['index']
                    else:
                        logfile.write("{}: Adding connection: source:{}, destination:{}\n".format(topmodule_name,self.wire_dict[topmodule_name+connection[1]]['source'],
                        gate_type_dict[gate_name]['index']))
                        self.wire_dict[topmodule_name+connection[1]]['destination'].append(gate_type_dict[gate_name]['index'])
                        edge_list.append(
                            (self.wire_dict[topmodule_name+connection[1]]['source'], gate_type_dict[gate_name]['index']))
            else:
                for connection in a_gate[1:]:
                    #if connect to intput
                    if (module_dict['Modules'][a_gate[0][1]]['IO'][connection[0]]['type_name'] =='Input'):
                        #if connect to power/ground
                        if(connection[1] in self.PowerAndGnd):
                            edge_list.append((self.wire_dict[connection[1]]['source'], module_dict['Modules'][a_gate[0][1]]['IO'][connection[0]]['index']))
                        else:
                            self.wire_dict[topmodule_name+connection[1]]['destination'].append(module_dict['Modules'][a_gate[0][1]]['IO'][connection[0]]['index'])
                            edge_list.append((self.wire_dict[topmodule_name+connection[1]]['source'],module_dict['Modules'][a_gate[0][1]]['IO'][connection[0]]['index']))
                    else:
                        self.wire_dict[topmodule_name+connection[1]]['source'] = module_dict['Modules'][a_gate[0][1]]['IO'][connection[0]]['index']
                        for dest in self.wire_dict[topmodule_name+connection[1]]['destination']:
                            edge_list.append((self.wire_dict[topmodule_name+connection[1]]['source'],dest))
                        
                    #if connect from output
                    
        logfile.write("test_final: {}, {}\n".format("1'b0",self.wire_dict["1'b0"]['destination']))
        module_dict['IO'] = {}
        module_dict['Gates'] = {}
        module_dict['IO'] = IO_dict
        module_dict['Gates'] = gate_type_dict
        module_dict['PowerAndGnd'] = {}
        module_dict['PowerAndGnd'] = self.PowerAndGnd
        Gx = nx.Graph()
        Gx.add_nodes_from(node_list)
        # print(edge_list)
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
            with open(savepath+'wire_dict.json', 'w') as fp:
                json.dump(self.wire_dict, fp)

            np.savetxt(savepath+topmodule_name+'_nodelist.txt',node_list,"%s",delimiter=",")
            np.savetxt(savepath+topmodule_name+'_A.txt',edge_list,"%s",delimiter=",")

            node_labels = np.zeros((len(node_list),1))
            readme = []
            node_labels,readme,current_idx = self.GetNodeLabelFromModuleDict(module_dict,readme,node_labels,starting_idx=0,logfile=logfile)
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

    def GetNodeLabelFromModuleDict(self, module_dict,readme,node_labels,starting_idx=0,logfile=None):
        current_idx = starting_idx
        idx=starting_idx
        for mod in module_dict['Modules']:
            logfile.write("starting idx:{};  idx adjust to:{}\n".format(starting_idx,idx))
            if(bool(module_dict['Modules'][mod]['Modules'])):
                logfile.write("found submodules in: {};  starting_idx is:{}\n".format(mod,idx))
                node_labels,readme,current_idx = self.GetNodeLabelFromModuleDict(module_dict['Modules'][mod],readme,node_labels,starting_idx=idx,logfile=logfile)
                idx=current_idx
            logfile.write("current_idx :{};  idx adjust to:{}\n".format(current_idx,idx))
            node_idx = [module_dict['Modules'][mod]['Gates'][g]['index'] for g in module_dict['Modules'][mod]['Gates']]
            node_idx += [module_dict['Modules'][mod]['IO'][g]['index'] for g in module_dict['Modules'][mod]['IO']]
            node_labels[node_idx]=idx
            readme.append((mod,idx))
            idx+=1
            current_idx=idx
        return node_labels,readme,current_idx

    def FPGAVerilogParser(self,filename,module_filename):
        module_file = open(module_filename, 'r') 
        lines = module_file.readlines() 
        module_lib = {}
        for l in lines:
            text = l.split(';')
            cell_type = text[0]
            module_lib[cell_type] ={}
            module_lib[cell_type]['input_ports'] = []
            module_lib[cell_type]['input_ports'] = text[3].split(',')
            module_lib[cell_type]['output_ports'] = text[4].split(',')
            # num_input = text[1]
            # num_output = text[2]
            # num_parms = text[3]

        with open(filename, 'rt') as fh:
            lines = fh.readlines()
        cell_list=[]
        for l in lines:
            text = l.split(';')
            fields = text[0].split(',')
            a_gate = []
            cell_type = fields[0]
            cell_name = fields[1]
            a_gate.append([cell_type,cell_name])
            counter = 1
            counter_freeze = False
            for f in fields[2:]:
                if('{' in f):
                    counter_freeze = True
                    f = f.replace("{","")
                    dst = f
                if('}' in f):
                    counter_freeze = False
                    f = f.replace("}","")
                    dst = f
                if(counter%2 != 0):
                    src = f
                else:
                    dst = f
                    a_gate.append([src,dst])
                if(not counter_freeze):
                    counter +=1
            cell_list.append(a_gate)
        with open('module_lib.json', 'w') as fp:
            json.dump(module_lib, fp)

        return module_lib,cell_list
    
    def FPGAtoGraph(self,filename,module_filename,vo_file,ports_to_exclude=[],wire_to_exclude=[],hierarchy_identifier='|'):
        module_lib,cell_list = self.FPGAVerilogParser(filename,module_filename)
        cell_dict = {}
        wire_dict = {}
        node_list = []
        idx = 0

        for a_cell in cell_list:
            # if is not a submodule
            node_list.append(idx)
            cell_type = a_cell[0][0]
            cell_name = a_cell[0][1]

            cell_dict[cell_name] = {}
            cell_dict[cell_name]['index'] =idx
            cell_dict[cell_name]['type'] = cell_type
            cell_dict[cell_name]['connections'] = {}
            for connection in a_cell[1:]:
                port_name = connection[0]
                wire_name = connection[1]
                if(port_name in module_lib[cell_type]['output_ports']):
                    if(not port_name in cell_dict[cell_name]['connections']):
                        cell_dict[cell_name]['connections'][port_name] = []
                    cell_dict[cell_name]['connections'][port_name].append(wire_name)

                    if(not wire_name in wire_dict):
                        # print(wire_name)
                        wire_dict[wire_name] = {}
                        wire_dict[wire_name]['destination'] = []
                    wire_dict[wire_name]['source_cell'] = cell_name
                    wire_dict[wire_name]['source_port'] = port_name
                
                elif(port_name in module_lib[cell_type]['input_ports']):
                    
                    if(not port_name in cell_dict[cell_name]['connections']):
                        cell_dict[cell_name]['connections'][port_name] = []
                    if(port_name in ports_to_exclude): #removing any clk and gnd connection
                        continue
                    cell_dict[cell_name]['connections'][port_name].append(wire_name)
                    if(not wire_name in wire_dict):
                        wire_dict[wire_name] = {}
                        wire_dict[wire_name]['destination'] = []
                    wire_dict[wire_name]['destination'].append([cell_name,port_name])
                    
                    # print(wire_dict[wire_name])
            idx += 1

        ## fix ram connection issue
        ref = open(vo_file, "r")
        mapping = {}
        for l in ref.readlines():
            if('assign' in l ):
                texts = l.split('=')
                wires = (texts[0].split('assign')[-1]).split('\\')[-1].split(' ')[:-1]
                wire_left = wires[0]
                for entry in wires[1:]:
                    wire_left += entry
                wires = texts[1].split('\\')[-1].split(' ')[:-1]
                wire_right = wires[0]
                # for entry in wires[1:]:
                #     wire_right += entry
                    
                if(wire_right in mapping):
                    mapping[wire_right].append(wire_left)
                else:
                    mapping[wire_right] = []
                    mapping[wire_right].append(wire_left)
        print("{} assigns found".format(len(mapping)))
        for wire in wire_dict:
            if wire in mapping:
                m_wires = mapping[wire]
                for m_wire in m_wires:
                    # if('q_b' in m_wire):
                    #     print("mapping {} to  {}".format(m_wire, wire))
                    if m_wire in wire_dict:
                        wire_dict[m_wire]['source_port'] = wire_dict[wire]['source_port']
                        wire_dict[m_wire]['source_cell'] = wire_dict[wire]['source_cell']
                

        #create node_list and edge_list from dictionaries

        node_list = []
        edge_list = []
        label_list = []
        label_dict = {}
        edge_label_list = []
        signal_collection = {}
        input_dict={}
        for wire in wire_dict:
            ## removing unwanted connections
            if(wire in wire_to_exclude):#['n92','GND','VCC','vcc','n216','n1','n4577','n4576','clk','gnd','u_sys_pll|altpll_component|_clk0','u_sys_pll|altpll_component|_clk1','u_sys_pll|altpll_component|_clk2']):
                print('Found unwanted signal:',wire)
                print('Removing...')
                if( not wire in signal_collection):
                    signal_collection[wire] = {}
                for cell_name in wire_dict[wire]['destination']:
                    hierarchy = cell_name[0].split(hierarchy_identifier)
                    signal_collection[wire] = count_hierarchy(signal_collection[wire],hierarchy,0)
                continue
            
            if('source_cell' in wire_dict[wire]):
                source = wire_dict[wire]['source_cell']
                for dst in wire_dict[wire]['destination']:
                    edge_list.append([cell_dict[source]['index'],cell_dict[dst[0]]['index']])
                    if('MULT' in wire):
                        edge_label_list.append(36)
                    else:
                        edge_label_list.append(1)
                    
            else:
                print("Found",wire,"has no source. It could be a input signal!\n")
                
                cell_name = wire
                input_dict[cell_name]=1
                cell_dict[cell_name] = {}
                cell_dict[cell_name]['index'] =idx
                cell_dict[cell_name]['type'] = 'Ext_Inputs'
                cell_dict[cell_name]['connections'] = {}
                for dst in wire_dict[wire]['destination']:
                    edge_list.append([idx,cell_dict[dst[0]]['index']])
                    if('MULT' in wire):
                        edge_label_list.append(36)
                    else:
                        edge_label_list.append(1)
                idx+=1
        label_index = 1
        for cell in cell_dict:
            node_list.append(cell_dict[cell]['index'])

            labels = cell.split(hierarchy_identifier)
            label_str = ''
            for l in labels: 
                if (not l in label_dict):
                    label_dict[l] = label_index
                    label_index += 1
                label_str = label_str + ';' + str(label_dict[l])
            label_list.append(label_str)
        with open(self.top_module_name+'/'+'cell_dict.json', 'w') as fp:
            json.dump(cell_dict, fp)
        with open(self.top_module_name+'/'+'label_dict.json', 'w') as fp:
            json.dump(label_dict, fp)
        with open(self.top_module_name+'/'+'wire_dict.json', 'w') as fp:
            json.dump(wire_dict, fp)
        with open(self.top_module_name+'/'+'signal_collection.json', 'w') as fp:
            json.dump(signal_collection, fp)
        with open(self.top_module_name+'/'+'input_dict.json', 'w') as fp:
            json.dump(input_dict, fp)
        with open(self.top_module_name+'/'+'wire_mapping.json', 'w') as fp:
            json.dump(mapping, fp)
        np.savetxt(self.top_module_name+'/'+self.top_module_name+'_node_labels.txt',label_list,"%s",delimiter=",")
        np.savetxt(self.top_module_name+'/'+self.top_module_name+'_nodelist.txt',node_list,"%s",delimiter=",")
        np.savetxt(self.top_module_name+'/'+self.top_module_name+'_A.txt',edge_list,"%s",delimiter=",")
        np.savetxt(self.top_module_name+'/'+self.top_module_name+'_edge_labels.txt',edge_label_list,"%s",delimiter=",")

        np.savetxt(self.top_module_name+'/'+self.top_module_name+'_graph_indicator.txt',np.zeros(len(node_list),dtype=np.intc),"%s",delimiter=",")
        np.savetxt(self.top_module_name+'/'+self.top_module_name+'_graph_labels.txt',np.zeros(1,dtype=np.intc),"%s",delimiter=",")
        return module_lib,node_list

    def gnlVerilogParser(self,filename,modulename,savepath='./'):
        f = open(filename, 'r') 
        lines = f.readlines() 

        input_port_list = []
        output_port_list = []
        gate_list = []
        module_list = {}
        read_begin = False

        node_idx = 0
        node_list = []
        edge_list = []
        wire_dict = {}
        gate_dict = {}
        for l in lines:
            if('#' in l):
                continue
            if('circuit' in l):
                read_begin = True
                text = l.split(' ')
                # modulename = text[1]
                continue
            if(('combinational' in l) or ('sequential' in l)):
                l = l.split('\n')[0]
                text = l.split(' ')
                current_module_name = text[1]
                module_list[current_module_name]={}
                module_list[current_module_name]['inputs'] = 0
                module_list[current_module_name]['outputs'] = 0
            if('end' in l):
                current_module_name = ''
            if(not read_begin):
                l = l.split('\n')[0]
                text = l.split(' ')
                if(text[0] == 'input'):
                    for g in text[1:]:
                        module_list[current_module_name]['inputs'] +=1
                if(text[0] == 'output'):
                    for g in text[1:]:
                        module_list[current_module_name]['outputs'] +=1
            
            if(read_begin):
                l = l.split('\n')[0]
                text = l.split(' ')
                gate_type = text[0]
                if(text[0] == 'input'):
                    for g in text[1:]:
                        input_port_list.append(g)
                        node_list.append(node_idx)
                        gate_dict[node_idx]={}
                        gate_dict[node_idx]['type'] = text[0]
                        if not g in wire_dict:
                            wire_dict[g]={}
                            wire_dict[g]['name'] = g
                            wire_dict[g]['destinations']=[]
                            wire_dict[g]['source'] = node_idx
                        node_idx+=1
                elif(text[0] == 'output'):
                    for g in text[1:]:
                        output_port_list.append(g)
                        # node_list.append(node_idx)
                        gate_dict[node_idx]={}
                        gate_dict[node_idx]['type'] = text[0]
                        if not g in wire_dict:
                            wire_dict[g]={}
                            wire_dict[g]['name'] = g
                            wire_dict[g]['destinations']=[]
                        # node_idx+=1
                elif(text[0] in module_list):
                    node_list.append(node_idx)
                    gate_dict[node_idx]={}
                    gate_dict[node_idx]['type'] = text[0]
                    gate_dict[node_idx]['inputs'] = []

                    n_input = module_list[text[0]]['inputs']
                    n_output = module_list[text[0]]['outputs']
                    for w in text[1:n_input+1]:
                        gate_dict[node_idx]['inputs'].append(w)
                        if not w in wire_dict:
                            wire_dict[w]={}
                            wire_dict[w]['name']=w
                            wire_dict[w]['destinations']=[]
                        wire_dict[w]['destinations'].append(node_idx)
                    for w in text[-n_output:]:
                        gate_dict[node_idx]['output'] = w
                        if not w in wire_dict:
                            wire_dict[w]={}
                            wire_dict[w]['name'] = w
                            wire_dict[w]['destinations']=[]
                        wire_dict[w]['source'] = node_idx

                    node_idx+=1
        for w in wire_dict:
            wire = wire_dict[w]
            print(wire)
            for dest in wire['destinations']:
                edge_list.append((wire['source'],dest))

        node_labels = np.zeros((len(node_list),1))
        graph_labels=[]
        graph_labels.append(0)
        graph_indicator = np.zeros((len(node_list),1))


        node_labels = extract_gnl_hierarchy(directory=savepath,top_modulename=modulename,node_label=node_labels)

        np.savetxt(savepath+'/'+modulename+'_nodelist.txt',node_list,"%s",delimiter=",")
        np.savetxt(savepath+'/'+modulename+'_A.txt',edge_list,"%s",delimiter=",")
        np.savetxt(savepath+'/'+modulename+'_node_labels.txt',node_labels,"%d",delimiter=",")
        np.savetxt(savepath+'/'+modulename+'_graph_labels.txt',graph_labels,"%d",delimiter=",")
        np.savetxt(savepath+'/'+modulename+'_graph_indicator.txt',graph_indicator,"%d",delimiter=",")

        with open('gate_dict.json', 'w') as fp:
            json.dump(gate_dict, fp)
        with open('wire_dict.json', 'w') as fp:
            json.dump(wire_dict, fp)
        
        
def extract_gnl_hierarchy(directory,top_modulename,node_label):

    filelist = os.listdir(directory)
    node_lib = {}
    submodule_list = {}
    top_module_gate_list = []
    pattern='*hnl'
    for f in filelist:
        if fnmatch.fnmatch(f, pattern):
            print(f)
            fid = open(directory+'/'+f, 'r') 
            lines = fid.readlines() 
            read_begin = False

            gate_type_list=[]
            for l in lines:
                if('#' in l):
                    continue
                if('circuit' in l):
                    read_begin = True
                    text = l.split(' ')
                    modulename = text[1]
                    continue
                if('end' in l):
                    current_module_name = ''
                if(read_begin):
                    
                    l = l.split('\n')[0]
                    text = l.split(' ')
                    if((text[0]=='input')or(text[0]=='output')or(text[0]=='end')):
                        continue
                    gate_type_list.append(text[0])

            if (os.path.splitext(f)[0] == top_modulename):

                top_module_gate_list = gate_type_list
            else:
                submodule_list[os.path.basename(f)]=gate_type_list
    num_input = len(node_label)-len(top_module_gate_list)
    for i in range(len(top_module_gate_list)):
        node_lib[i] = 0
    for idx,s in enumerate(submodule_list):
        gate_list = submodule_list[s]
        index = find_index(top_module_gate_list,gate_list)
        for i in range(index,index+len(gate_list)):
            node_lib[i+num_input] = idx + 1 
            node_label[i+num_input] = idx + 1
    
    # print(stats)
    with open(directory+'/node_lib.json', 'w') as fp:
        json.dump(node_lib, fp)
    return node_label     
def find_index(a,b):
    for i in range(len(a)):
        if (b == a[i:len(b)+i]):
            return i
    return 0
def count_hierarchy(stats_dict,label_txt,index):
    if(index==len(label_txt)):
#         print(stats_dict)
#         print('test')
        return stats_dict
#     print(stats_dict)
#     print(label_txt[index])
    if(not label_txt[index] in stats_dict):
        stats_dict[label_txt[index]] = {}
        stats_dict[label_txt[index]]['count'] = {}
        stats_dict[label_txt[index]]['count'] =1
    else:
        stats_dict[label_txt[index]]['count'] +=1
    stats_dict[label_txt[index]] = count_hierarchy(stats_dict[label_txt[index]],label_txt,index+1)
    return stats_dict

def getkey(dictionary,search_value):
    for key, value in dictionary.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if(value == ''):
            print(key,value)
        elif (int(value) == int(search_value)):
            return key 

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


