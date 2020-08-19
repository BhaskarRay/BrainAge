addpath(genpath('/trdapps/linux-x86_64/matlab/toolboxes/spm12'));
%Add gift toolbox

k_com=spm_read_vols(spm_vol('626sub_agg__component_ica_.nii'));
[x,y,z,c] = size(k_com);
k_com = reshape(k_com,x*y*z,c);

b_com=spm_read_vols(spm_vol('SpatialTemporalRegression_agg__component_ica_.nii'));
[xx,yy,zz,cc] = size(b_com);
b_com = reshape(b_com,xx*yy*zz,cc);

[r]=corr(b_com,k_com);

