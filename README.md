# util_funcs
## Usage:
run perl conversion script: ```perl 'FPGA_Netlist_Txt_Conversion.pl' 'Data\try_vo.vo' 'Data\try_vo.txt'``` <br>
run ReadFPGA.ipyn


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
