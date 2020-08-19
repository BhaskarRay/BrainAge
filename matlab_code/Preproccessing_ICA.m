clc;
clear all;
close all;

%% loading imaging --------
imgpath='/export/mialab/users/kduan/ADHD/Adult_IMG_NeuroIMAGE_IMpACT/';% change to your path
imgfiles = dir([imgpath,'smwc1*nii']);% changed to your files
inum = length(imgfiles);
filename = cell(inum,1);
for i=1:inum
    filetemp=imgfiles(i).name;
    filename{i} = strcat(imgpath,filetemp);
    sMRIdata(:,:,:,i)=spm_read_vols(spm_vol(filename{i}));  
end

%% QC on imaging data 
[x,y,z,sub] = size(sMRIdata);
sMRIdata1 = reshape(sMRIdata,x*y*z,sub);
meansMRI = mean(sMRIdata1,2);
[r] = corr(meansMRI,sMRIdata1);
figure;plot(r,'.');
removind = find(r<0.8);
sMRIdata1(:,removind)=[];

%%%%%generate mask
meansMRI = mean(sMRIdata1,2);
sind_mask = find(meansMRI>0.2);
sMRIdata_n = sMRIdata1(sind_mask,:);

%%%%%%%correct age,gender and site effects
%use controls' age and gender infomation estimate coefficients for age and gender; use all subjects' site info to estimate the coefficient for site, and then regress out age,gender and site effects for all subjects;
%need controls' sMRI data(control_sMRIdata_n),age(control_subj_age) and gender(control_subj_gender); as well as all subjects' sMRI data(sMRIdata_n), age(subj_age),gender(subj_gender) and site(adult_subj_site_var) infomation.
control_sMRIdata_n = sMRIdata_n(:,control_index);
X_control = [control_subj_age,control_subj_gender];
[vv mm] = size(sMRIdata_n);
sMRIdata_n_c = zeros(vv,mm);
for i = 1:size(control_sMRIdata_n,1)
    stats1 = regstats(control_sMRIdata_n(i,:)',X_control,'linear',{'tstat'});
    beta_age_tmp = stats1.tstat.beta(2);
    beta_gender_tmp = stats1.tstat.beta(3);
    stats2 = regstats(sMRIdata_n(i,:)',adult_subj_site_var,'linear',{'tstat'});
    beta_site_tmp = stats2.tstat.beta(2);
    sMRIdata_n_c(i,:) = sMRIdata_n(i,:)'-beta_age_tmp*subj_age-beta_gender_tmp*subj_gender-beta_site_tmp*adult_subj_site_var;
end

%%%%%%%%%%%%correct age,gender and site effects
%use all children's age, gender and site infomation to estimate regression coefficients 
subj_age;
subj_gender;
subj_site;
X = [subj_age,subj_gender,subj_site];
[vv mm] = size(sMRIdata_n);
sMRIdata_n_c = zeros(vv,mm);
for i = 1:size(sMRIdata_n,1)
    stats1 = regstats(sMRIdata_n(i,:)',X,'linear',{'tstat'});
    beta_age_tmp = stats1.tstat.beta(2);
    beta_gender_tmp = stats1.tstat.beta(3);
    beta_site_tmp = stats2.tstat.beta(4);
    sMRIdata_n_c(i,:) = sMRIdata_n(i,:)'-beta_age_tmp*subj_age-beta_gender_tmp*subj_gender-beta_site_tmp*subj_site;
end

% %%%%write age&gender corrected data into niffti file
% anatomy_vol = spm_vol('Y:\kduan\ADHD\Adult_IMG_NeuroIMAGE_IMpACT\smwc1m19-4-0003_t1_structural_20101106_09_0176.nii');
% anatomy = spm_read_vols(anatomy_vol);  
% [x,y,z] = size(anatomy);
% 
% for i=1:size(sMRIdata_n_c,2)
%      subj_sm_cor_niffti = zeros(x*y*z,1);
%      subj_sm_cor_niffti(sind_mask(:)) = sMRIdata_n_c(:,i);
%      subj_sm_cor_niffti = reshape(subj_sm_cor_niffti,x,y,z);
%      VV = anatomy_vol;
%      vv_name_tmp = strcat('smwc1m',adult_subj_info(i,1));
%      VV.fname = strcat(vv_name_tmp,'_t1_corr.nii');
%      spm_write_vol(VV,subj_sm_cor_niffti);
% end

%% estimate componeent nubmer 
[b1, mdlVal, aicVal] = ica_fuse_estimate_dimension(sMRIdata_n_c, [6 6 6]);

%%%%run ICA multiple times and choose the result from the most stable run
% data - Subjects x voxels
% numOfIC - Number of components desired
% num_ica_runs - Number of times ICA is run
% algorithm - 1 - Infomax, 2 - Fast ica
% A - Mixing matrix
% icasig - Sources
data = sMRIdata_n_c';
numOfIC = b1;
num_ica_runs = 10;
algorithm = 1;
[A, W, icasig, iq, metric_Q,sR] = myIcasso(data, numOfIC, num_ica_runs, algorithm);

%%%%plot components
figure;
plotfMRI(icasig(:,:),sind_mask, 2, 1);






