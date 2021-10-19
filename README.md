# util_funcs
## Usage:
1. run perl conversion script: ``` perl 'FPGA_Netlist_Txt_Conversion.pl' './fft_64/fft_64.vo' './fft_64/fft_64_vo.txt'``` <br>
1. create  ```module_name```folder
1. run: ```python read_fpga.py --modulename fft_64 --netlist_filename ./fft_64/fft_64_vo.txt  --vo_file ./fft_64/fft_64.vo --lib_filename ./cell_libs/stratixiii_cell_library.txt --o ./```<br>
1. change ```module_node_label.txt``` to ```module_node_label_bk.txt``` (to avoid script read node labels).<br>
1. 

# TODO
verilog_utils:<br>
return a dictionary with all gate mapped.<br>
- Toplevel:
  - gates:
    - gate1:(gate_name)
      - "index"
      - "gate_name"
      - "type"
      - "type_name"
    - gate2:
  - modules:
    - module1:
      - IO:
        - input1:
        - output1:
      - Gates:
        - gate1:
          - "index"
          - "name"
          - "type"
          - "type_name"
        - gate2:...
    - module2:...
  
write a log file to same path of circuit folder.
