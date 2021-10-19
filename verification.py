# this program is to verify the number of modules whether match the processed file

import os 
# original exported netlist
file_name = './fft_64/fft_64.vo'
file_object = open(file_name,'r+')
# processed netlist
file_name2 = './fft_64/fft_64_vo.txt'
module_count = 0
module_list = []


for line in file_object:
    if ("// Location:") in line:
        module_count += 1
        next_line = next(file_object)
        module_name = next_line.split(' ')[0]
        module_list.append(module_name)

module_set = list(set(module_list))

# read the original netlist file, got the module set and No. of modules in total
print('Read orignal netlist finished!')
print(f'There are {module_count} modules in total, {len(module_set)} different modules, the following is the list:\n{module_set}\n\n')

# use iterial method count the processed file
def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)
module_count2 = iter_count(file_name2)

if module_count2 == module_count:
    print('The modules included in the processed netlist file MATCH the original')
else:
    print('The modules included in the processed netlist file NOT MATCH the original')