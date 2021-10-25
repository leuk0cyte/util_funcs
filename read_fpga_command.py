import os 

module_name = 'fft_64'
cell_lib_filename = './cell_libs/stratixiii_cell_library.txt'

command = f"perl './FPGA_Netlist_Txt_Conversion.pl' './{module_name}/{module_name}.vo' './{module_name}/{module_name}_vo.txt'"

# run the perl script
os.system(command)

# replace the '.\' to ',' in the perl's output file
txt_file_obj = open(f'./{module_name}/{module_name}_vo.txt','+r')
txt_file_cont = txt_file_obj.read()
txt_file_cont = txt_file_cont.replace(',\\',',')
txt_file_obj.seek(0,0)
txt_file_obj.write(txt_file_cont)
txt_file_obj.truncate()

# run the read_fpga script
command =  f'python read_fpga.py --modulename {module_name} --netlist_filename ./{module_name}/{module_name}_vo.txt  --vo_file ./{module_name}/{module_name}.vo --lib_filename {cell_lib_filename} --o ./'
os.system(command)

# verify the number of modules in the .vo file and .txt file
import verification