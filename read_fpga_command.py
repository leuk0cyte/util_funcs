import os 
module_name = 'fir'
cell_lib_filename = './cell_libs/stratixiii_cell_library.txt'
command =  f'python read_fpga.py --modulename {module_name} --netlist_filename ./{module_name}/{module_name}_vo.txt  --vo_file ./{module_name}/{module_name}.vo --lib_filename {cell_lib_filename} --o ./'

os.system(command)