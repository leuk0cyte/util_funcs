import sys
from util_funcs import graph_utils
from util_funcs import verilog_utils
import argparse

"""usage: 
    python read_fpga.py --modulename [modulename] --netlist_filename path-to-netlist --o path-to-output-folder"""

def arg_parse():
    parser = argparse.ArgumentParser(description="arguments parser")

    parser.add_argument("--modulename",
                        dest="modulename",
                        default='fft',
                        help="The experiment to reproduce")
    parser.add_argument("--netlist_filename",
                        dest="netlist_filename",
                        default='fft',
                        help="run spectral clustering")
    parser.add_argument("--lib_filename",
                        dest='lib_filename',
                        default='cell_library.txt',
                        help="run_VGAE")
    parser.add_argument("--port_to_exclude",
                        dest="port_to_exclude",
                        default=[],
                        help="run MincutPool")
    parser.add_argument("--wire_to_exclude",
                        dest="wire_to_exclude",
                        default=[],
                        help="run diffpool")
    parser.add_argument("--o",
                        dest="output",
                        default=[],
                        help="run diffpool")

    return parser.parse_args()


# module_lib,cell_list = reader.FPGAtoGraph(netlist_filename,lib_filename)
# port_to_exclude = ['clk','clk0','clk1','inclk']
# wire_to_exclude = ['vcc','VCC','GND','gnd','inclk','clk1','clk0','clk2','clk3','vcc','n216','n4577','n4576']

if __name__ == '__main__':
    args = argparse()
    reader = verilog_utils.verilog_reader(args.modulename,
                                          args.netlist_filename)
    module_lib, cell_list = reader.FPGAtoGraph(args.netlist_filename,
                                               args.lib_filename,
                                               args.vo_file,
                                               args.port_to_exclude,
                                               args.wire_to_exclude,
                                               savepath=args.output,
                                               hierarchy_identifier='|')
