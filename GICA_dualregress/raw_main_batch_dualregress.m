%% batch used to run the main gICA on all the data

% performance setting: Maximize performance
perfType = 1;
% Group ICA using ICASSSO
which_analysis = 2;
icasso_opts.sel_mode = 'bootstrap';
icasso_opts.num_ica_runs = 10;
% 1 means infomax, 2 means fastICA, etc.
algoType = 1;
%% selecting data
dataSelectionMethod = 4;
base_dir = '/data/mialab/users/bray14/FD_FMRI/FMRI_DATA/'; %%change to your data path

files_name = dir([base_dir '*.nii']);

for i=1:size(files_name,1)
    input_data_file_patterns{i,1}=[base_dir files_name(i).name];
end

% TR = cell2mat(subid_age_gender_TR(:,4));

dummy_scans = 4; % Enter no. of dummy scans to exclude from the group ICA analysis. If you have no dummy scans leave it as 0.
keyword_designMatrix = 'no'; % no design matrix is specified

clear base_dir 
%% output dir info
outputDir = '/data/mialab/users/bray14/fd_fmri_3rdAttempt2/';%%change to your output directory
prefix = 'SpatialTemporalRegression';
maskFile = [];
%% specify preprocessing type
% Variance normalization
preproc_type = 4;
%% dimension reduction (PCA)
group_pca_type = 'subject specific';
pcaType = 'SVD';
doEstimation = 0; 

numReductionSteps = 2;
% number of pc to reduce each subject down to at each reduction step
% The number of independent components the will be extracted is the same as 
% the number of principal components after the final data reduction step.  
numOfPC1 = 110;
numOfPC2 = 100;
%% backreconstruction type
% regular
backReconType = 2;
%%BACKRECON_DEFAULT: Back-reconstruction default.
%??? 1 - Regular
%??? 2 - Spatial-temporal Regression
%??? 3 - GICA3
%??? 4 - GICA

%% Scale the Results. Options are 0, 1, 2
% 0 - Don't scale
% 1 - Scale to Percent signal change
% 2 - Scale to Z scores
scaleType = 2;