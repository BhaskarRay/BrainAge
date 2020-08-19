clc
clear
addpath(genpath('/trdapps/linux-x86_64/matlab/toolboxes/GroupICATv4.0b/icatb')); %Add gift toolbox
icatb_read_batch_file('/data/mialab/users/bray14/GICA_dualregress/raw_main_batch_dualregress.m');%load parameter file
icatb_batch_file_run('/data/mialab/users/bray14/GICA_dualregress/raw_main_batch_dualregress.m');% run 
