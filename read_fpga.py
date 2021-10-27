import sys
import graph_utils
import verilog_utils
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
    parser.add_argument("--vo_file",
                        dest="vo_file",
                        default='fft',
                        help="vo_file")
    parser.add_argument("--lib_filename",
                        dest='lib_filename',
                        default='--',
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
# port_to_exclude = ['vcc','gnd','clk','reset_n','devclrn','devpor']
# wire_to_exclude = ['vcc','gnd','clk','reset_n','devclrn','devpor','clk~inputclkctrl_outclk','reset_n~inputclkctrl_outclk']

if __name__ == '__main__':
    args = arg_parse()
    reader = verilog_utils.verilog_reader(args.modulename,
                                          args.netlist_filename)
    module_lib, cell_list = reader.FPGAtoGraph(args.netlist_filename,
                                               args.lib_filename,
                                               args.vo_file,
                                               savepath=args.output,
                                               ports_to_exclude=args.port_to_exclude,
                                               wire_to_exclude=args.wire_to_exclude,
                                               hierarchy_identifier='|')
