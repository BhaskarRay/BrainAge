yourpath_fd = 'C:/Users/bray14/Downloads/FD/';
% Get all the subfolders
ContentInFold_fd = dir(yourpath_fd);
SubFold_fd = ContentInFold_fd([ContentInFold_fd.isdir]); % keep only the directories
% Loop on each folder
destpath_fd = 'C:/Users/bray14/Downloads/';
newFolder2 = fullfile(destpath_fd,'FD_all_subjects');
for i = 3:length(SubFold_fd)% start at 3 to skip . and ..
    tempPath = fullfile(yourpath_fd,SubFold_fd(i).name,'func','task-rest_bold','\\');
    %rp_sub-600049785291_task-rest_bold.txt
    txtfile_fd = dir([tempPath,'rp_sub-*txt']);
    outputBaseFileName_fd = strcat('fwd','_rp_',SubFold_fd(i).name,'_task-rest_bold.txt');
    inputFullFileName_fd = fullfile(tempPath, txtfile_fd.name);
    outputFullFileName_fd = fullfile(newFolder2, outputBaseFileName_fd);
    outputFullFileNameS_folder_fd = fullfile(tempPath, outputBaseFileName_fd);
    %cd(temp)
    if isfile(inputFullFileName_fd)
        M=dlmread(inputFullFileName_fd);
        temp= M(:,4:6);
        temp=50*temp;
        M(:,4:6)=temp;
        dts=diff(M);

        dts=[
                zeros(1,size(dts,2)); 
                dts
                ];  % first element is a zero, as per Power et al 2014
        fwd=sum(abs(dts),2);
        dlmwrite(outputFullFileName_fd,fwd)
        dlmwrite(outputFullFileNameS_folder_fd,fwd)
    else
    fid = fopen(fullfile(newFolder2, 'Log_Missing_data_for_subjects.txt'), 'a');
    if fid == -1
    error('Cannot open log file.');
    end
    fprintf(fid, '%s\n',SubFold_fd(i).name);
    fclose(fid);
    %fprintf('%g Not exist for\n', SubFold_fd(i).name);
    end
    
    
end







  

