import os 

module_name = 'fft_4096' 
fpga_family = 'stratixiii'
cell_lib_filename = f'./cell_libs/{fpga_family}_cell_library.txt'      

port_to_exclude = "['vcc','gnd','devclrn','devpor','devoe','clk','reset_n']"
wire_to_exclude = "['vcc','gnd','devclrn','devpor','devoe','clk','reset_n','clk~input_o','clk~inputclkctrl_outclk','reset_n~input_o','reset_n~inputclkctrl_outclk']"
# devclrn,devpor;     // device wide clear/reset

# this function is to verify the number of modules whether match the processed file
def verification(object_name):
    # original exported netlist
    file_name = f'./{object_name}/{object_name}.vo'
    file_object = open(file_name,'r+')
    # processed netlist
    file_name2 = f'./{object_name}/{object_name}_vo.txt'
    module_count = 0
    module_list = []


    for line in file_object:
        if ("// Location:") in line:
            next_line = next(file_object)
            module_name = next_line.split(' ')[0]
            module_list.append(module_name)

    module_set = list(set(module_list))

    # read the original netlist file, got the module set and No. of modules in total
    print('Read orignal netlist finished!')
    print(f'There are {len(module_list)} modules in total, {len(module_set)} different kinds of module, the following is the list:\n{module_set}\n\n')

    # use iterial method count the processed file
    def iter_count(file_name):
        from itertools import (takewhile, repeat)
        buffer = 1024 * 1024
        with open(file_name) as f:
            buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
            return sum(buf.count('\n') for buf in buf_gen)
    module_count2 = iter_count(file_name2)

    if module_count2 == len(module_list):
        print('The modules included in the processed netlist file MATCH the original')
    else:
        print('The modules included in the processed netlist file NOT MATCH the original')


if __name__ == '__main__':
    command = f"perl './FPGA_Netlist_Txt_Conversion.pl' './{module_name}/{module_name}.vo' './{module_name}/{module_name}_vo.txt'"

    # run the perl script
    os.system(command)

    # replace the ',\' to ',' in the perl's output file
    txt_file_obj = open(f'./{module_name}/{module_name}_vo.txt','+r')
    txt_file_cont = txt_file_obj.read()
    txt_file_cont = txt_file_cont.replace(',\\',',')
    txt_file_cont = txt_file_cont.replace(',!\\',',')
    txt_file_obj.seek(0,0)
    txt_file_obj.write(txt_file_cont)
    txt_file_obj.truncate()

    # run the read_fpga script
    command =  f'python read_fpga.py --modulename {module_name} --netlist_filename ./{module_name}/{module_name}_vo.txt  --vo_file ./{module_name}/{module_name}.vo --lib_filename {cell_lib_filename} --port_to_exclude {port_to_exclude} --wire_to_exclude {wire_to_exclude} --o ./'
    os.system(command)

    os.rename(f'./{module_name}/{module_name}_node_labels.txt',f'./{module_name}/{module_name}_node_labels_bk.txt')

    # verify the number of modules in the .vo file and .txt file
    verification(module_name)