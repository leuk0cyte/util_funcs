#!/usr/bin/perl
#======================================================================
#Created by Song Jingsi @ 23/Sep/19
#this is the program that converts .vo/vqm file to .txt netlist
#Input:
#FPGA netlist file (.vo)
#
#Output:
#text file (.txt)
#----------------------------------------------------------------------
#Change Log:
##25/Sep/2020
#Duplicate the bus signals when meet '{}'. 
#eg. clk({gnd, gnd, gnd, n4577})  ----> clk,gnd,clk,n4577
#
#----------------------------------------------------------------------
#or in matlab command window:
#perl('FPGA_Netlist_Txt_Conversion.pl', 'Data\try_vo.vo', 'Data\try_vo.txt');
#======================================================================
#get input argument
$verilog_name= $ARGV[0];
$text_name= $ARGV[1];

#read verilog file
open(INFILE, $verilog_name) or die "could not open $verilog_name: $!\n";
#write out
#open(OUTFILE, ">$text_name");

$/=undef;
$_=<INFILE>;

$_ =~ s/\/\/.*[^\n\r]*?//g; # delete comments with //
# $_ =~ s/^.*?stratix_lcell\s*(.*)$/$1/mg; # delete all lines before 'begin'
$_ =~ s/[\n\r]//g; # delete new line

#processing: # translate vqm to txt
@lines=split(';', $_);

#==========Start: Deal with assian with inverter ============
local %hash_assign_inv=();
foreach (@lines)
{	
	if($_ =~ /^\s*assign/){
		my($wire_left, $wire_right) = /^\s*assign\s*(\S+)\s*\=\s*(\S+)\s*/;
		if($wire_right =~ /^\s*\~.+?/){
			$wire_right =~ s/^\~//g;
			$hash_assign_inv{$wire_left} = $wire_right;}
		#print "1:$wire_left ,2:$hash_assign_inv{$wire_left}\n";	
	}#end if $_
}
#==========End: Deal with assian with inverter ============

$query = undef;
$param = undef;
local %hash_Group=();
$lcell_mark = 0;
foreach (@lines)
{	
	# print "Each line: $_\n";
	#if($_ !~ /^\s*module|^\s*(in|out)put|^\s*inout|^\s*wire|^\s*assign|^\s*endmodule|^\s*defparam|^$/)
	if($_ =~ /^\s*stratix_\S+/)
    {	
		#print "cell is $_\n";
		$lcell_mark =1;
		# save cell name, inst name and connections
        my($cell_name, $inst_name, $connection) = /^\s*(\S+)\s+(\S+)\s+\((.*)\)/;
		
		$cell_name=~s/\s+//g; # delete extra space in cell_name
        $inst_name=~s/\s+//g;
		#print "connection:$connection\n"; 
		my @connection_pairs= $connection =~ m/,?\s*(.*?\))\s*/g;# delete verilog syntax
		#============ Start to deal with the wires in .vqm file ================
		$current_pair = undef;
		@connection_pairs_new = ();
		foreach $pair (@connection_pairs){
			$pair =~ s/ *\.(\S+?)\(\s*(.*?)\)$\s*/$1,$2/g;
			$port_name = $1;
			$wire_name = $2;
			#print "pair:$pair\n";
			#print "port_name:$port_name\n";
			#print "wire_name:$wire_name\n";
			#=======Deal with Bus wire===============
			if  ($wire_name =~ /^\s*\{.+?/){
				$wire_name =~ s/\{//;
				$wire_name =~ s/\}//;
				my @bus_wires =split(',', $wire_name);
				local %seen =();#for unique bus signal eg..clk({gnd, gnd, gnd, n4577}),
 
				foreach $single_bus (@bus_wires){
					$single_bus=~ s/(^\s+|\s+$)//g;#remove the space 
					#print "single_bus:$port_name,$single_bus\n";
					unless ( exists $seen{$single_bus} ) {
						$duplicate_bus_pair = join(",",$port_name, $single_bus);
						push(@connection_pairs_new, $duplicate_bus_pair);
						$seen{$single_bus}  =1;
						#print "push wire:$port_name,$single_bus\n";
					}
					
				}

				next; #skip the steps below
			}
			
			#=======Remove empty wire ==============
			if ($wire_name =~ /^\s*$/){
				print "Remove an port with empty wire : Delete the port: '$port_name' in cell $cell_name $inst_name.\n";				
				next;
			}
			#=======Remove inverter with '!'=========
			if ($wire_name =~ /^\s*\!.+?/){
				print "Remove an Inverter : Delete the '!' of the wire: '$wire_name' in cell $cell_name $inst_name.\n";				
				$wire_name =~ s/\!//;
			}
			
			#=======Replace inverter with assign '~'==========

			if (exists ($hash_assign_inv{$wire_name}) ){			
				print "Replace an Inverter by assign: Replace the wire '$hash_assign_inv{$wire_name}' with '$wire_name' in cell $cell_name $inst_name .\n";
				$wire_name = $hash_assign_inv{$wire_name};
			}
			
			$current_pair = join(",",$port_name, $wire_name);
			push(@connection_pairs_new, $current_pair);
			
			# print "current_pair:$current_pair\n";
		}#end foreach $pair
		#============End to deal with the wires in .vqm file ================
		$connection = join(",",@connection_pairs_new);
		# print "connection: $connection\n";
		$query=join(",", $cell_name, $inst_name, $connection); 
		$query=~s/\s+//g; # delete extra space, because of empty wire/unuse this line
		# print "query:$query\n";	
		# push(@netlists, $_); # save them into a new array
		$param = undef;
		if (exists( $hash_Group{$query} )){print "This cell is the same with previous one :$inst_name ！\n "}else{$hash_Group{$query} = 0;}
		
	}elsif(($lcell_mark >= 1)&&($_ =~ /^\s*defparam/)){
		$_ =~ s/^\s*defparam.*\.(.*?)\s*\=\s*(.*?)/$1,$2/g;		
		#$temp = $1.','.$2;
		#print "para pair is $_\n";
		if($param eq "")
		{
			$param = "$_";
			#print "temp1 is $temp\n";
			#print "param1 is $param\n";	
		}else{
			$param =join(",",$param, $_); 
			#print "temp2 $temp\n";
			#print "param2 is $param\n";
		}
		$hash_Group{$query} = $param;
		$lcell_mark = $lcell_mark +1;
	}else{
		$lcell_mark = 0;
		# print "Unknown line: $_\n";
	}#end if... elsif

	
}

#output after cell_name error checking
open(OUTFILE, ">$text_name");
foreach $key (keys (%hash_Group) ){
	$line = join(";",$key,$hash_Group{$key});
	print OUTFILE $line, "\n";
	
}
$size = scalar keys %hash_Group;
print "====== Finish Netlist to Txt conversion! $size numbers of stratix_lcell are found. ======\n";


close(OUTFILE);
close(INFILE);
#processing: delete syntax and write out

