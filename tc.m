yourpath_fd = '/data/mialab/users/bray14/fd_fmri_3rdAttempt2/';
s1='SpatialTemporalRegression_ica_c';
s3='-1.mat';
for i = 1:1113
    
    s2 = int2str(i);
    sf = strcat(s1,s2,s3);
    
    tempPath = fullfile(yourpath_fd,sf);
    fid = fopen('TC_filename.txt', 'a');
    fprintf(fid, '%s\n',tempPath);
    fclose(fid);
    
end