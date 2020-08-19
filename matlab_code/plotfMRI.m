function plotfMRI(icasig_tmp,MASK,threshold, anatomy)

if nargin<3
    threshold =2;
end

if anatomy==1
    anatomy_vol = spm_vol('/home/users/bray14/template/ch2better.nii'); %%change to your anatomy image
    anatomy = spm_read_vols(anatomy_vol);  
end

numOfPC=size(icasig_tmp,1);
[h,w,l] = size(anatomy);

for i=1:numOfPC
    MRIimg=zeros(h*w*l,1);
    MRIimg(MASK(:))=zscore(icasig_tmp(i,:));% zscore function has the effect of normalization
    MRIimg = reshape(MRIimg,h,w,l);
    VV = anatomy_vol;
    vv_name_tmp = strcat('Component',num2str(i));
    VV.fname = strcat(vv_name_tmp,'.nii');
    h1=figure;subplot('Position',[0.05, 0.05, 0.9,0.9])
    make_composite(MRIimg(1:1:end,1:1:end,4:2:end-8),anatomy(1:end,1:end,4:2:end-8),threshold,'nearest');title(['fMRI component:',num2str(i)]);
end
