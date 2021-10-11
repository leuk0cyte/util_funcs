# netlist_filename = 'netlist_divided_test.txt'
import sys
from util_funcs import graph_utils
from util_funcs import verilog_utils


modulename = 'fft'
netlist_filename = '{}/{}_vo.txt'.format(modulename,modulename)
vo_file = '{}/{}.vo'.format(modulename,modulename)
lib_filename = 'cell_library.txt'

reader = verilog_utils.verilog_reader(modulename,netlist_filename)
# module_lib,cell_list = reader.FPGAtoGraph(netlist_filename,lib_filename)


# port_to_exclude = ['clk','clk0','clk1','inclk']
port_to_exclude=[]
# wire_to_exclude = ['vcc','VCC','GND','gnd','inclk','clk1','clk0','clk2','clk3','vcc','n216','n4577','n4576']
wire_to_exclude=[]
module_lib,cell_list = reader.FPGAtoGraph(netlist_filename,lib_filename,vo_file,port_to_exclude,wire_to_exclude,hierarchy_identifier='|')
