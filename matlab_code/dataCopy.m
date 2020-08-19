clc;
clear all;
close all;

% Reading data from subfolder
yourpath = '/data/mialab/UPENN/Processing_version1/';
% Get all the subfolders
ContentInFold = dir(yourpath);
SubFold = ContentInFold([ContentInFold.isdir]); % keep only the directories
% Loop on each folder
destpath = '/home/users/bray14/';
newFolder = fullfile(destpath,'Processing_version1_dataset');
for i = 3:length(SubFold)% start at 3 to skip . and ..
    temp = fullfile(yourpath,SubFold(i).name,'T1','NII','\\');
    imgfile = dir([temp,'Sm6mwc1*nii']);
    cd(temp)
    copyfile (imgfile.name, newFolder);
end