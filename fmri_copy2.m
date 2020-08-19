clc;
clear all;
close all;

% Reading data from subfolder
%yourpath = '/data/mialab/UPENN/AUTO_ANALYSIS_BIDS/version1/derivatives/';
yourpath = '/data/mialab/UPENN/AUTO_ANALYSIS_BIDS/version2_tmp/derivatives/';
% Get all the subfolders
ContentInFold = dir(yourpath);
SubFold = ContentInFold([ContentInFold.isdir]); % keep only the directories
% Loop on each folder
destpath = '/data/mialab/users/bray14/FD_FMRI/';
newFolder = fullfile(destpath,'mp_files');
for i = 3:length(SubFold)% start at 3 to skip . and ..
    
    
    temp = fullfile(yourpath,SubFold(i).name,'func','task-rest_bold','//');
    %imgfile = dir([temp,'swasub-*nii']);
	imgfile = dir([temp,'rp_sub-*txt']);
	
    %outputBaseFileName = strcat('sm6mwc1','sub-',SubFold(i).name,'_T1.nii');
    inputFullFileName = fullfile(temp, imgfile.name);
    outputFullFileName = fullfile(newFolder, imgfile.name);
    
    if exist(inputFullFileName,'file')==2

        %cd(temp)
        copyfile (inputFullFileName,outputFullFileName);
        
    else
    fid = fopen(fullfile(newFolder, 'Log_Missing_data_for_subjects.txt'), 'a');
    if fid == -1
    error('Cannot open log file.');
    end
    fprintf(fid, '%s\n',SubFold(i).name);
    fclose(fid);
    end

    
end