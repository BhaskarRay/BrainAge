clc;
clear all;
close all;

yourpath_fd = '/data/mialab/UPENN/AUTO_ANALYSIS_BIDS/version2/derivatives/';
% Get all the subfolders
ContentInFold_fd = dir(yourpath_fd);
SubFold_fd = ContentInFold_fd([ContentInFold_fd.isdir]); % keep only the directories
% Loop on each folder
destpath_fd = '/data/mialab/users/bray14/';
newFolder2 = fullfile(destpath_fd,'fd_version2_r');
newFolder3 = fullfile(destpath_fd,'fd_version_2_r_exclude');
for i = 3:length(SubFold_fd)% start at 3 to skip . and ..
    tempPath = fullfile(yourpath_fd,SubFold_fd(i).name,'func','task-rest_bold','//');
    %rp_sub-600049785291_task-rest_bold.txt
    txtfile_fd = dir([tempPath,'rp_sub-*txt']);
    outputBaseFileName_fd = strcat('fwd_',SubFold_fd(i).name,'.txt');
    outputBaseFileName_temp_fd = strcat('temp_fwd_',SubFold_fd(i).name,'.txt');
    outputBaseFileName_fd_num = strcat('fd_num_',SubFold_fd(i).name,'.txt');
    outputBaseFileName_fd_three = strcat('fd_mean_percentage',SubFold_fd(i).name,'.txt');
    
    outputBaseFileName_temp_fd_num = strcat('temp_fd_num_',SubFold_fd(i).name,'.txt');
    outputBaseFileName_temp_fd_three = strcat('temp_fd_mean_percentage',SubFold_fd(i).name,'.txt');
    
    
    inputFullFileName_fd = fullfile(tempPath, txtfile_fd.name);
    outputFullFileName_fd = fullfile(newFolder2, outputBaseFileName_fd);
    outputFullFileName_temp_fd = fullfile(newFolder3, outputBaseFileName_temp_fd);
    outputFullFileName_fd_three=fullfile(newFolder2, outputBaseFileName_fd_three);
    outputFullFileName_fd_num=fullfile(newFolder2, outputBaseFileName_fd_num);
    
    outputFullFileName_temp_fd_three =fullfile(newFolder3, outputBaseFileName_temp_fd_three);
    outputFullFileName_temp_fd_num =fullfile(newFolder3, outputBaseFileName_temp_fd_num);
    
    %outputFullFileNameS_folder_fd = fullfile(tempPath, outputBaseFileName_fd);
    %outputFullFileNameS_folder_temp_fd = fullfile(tempPath, outputBaseFileName_temp_fd);
    %outputFullFileNameS_folder_fd_three=fullfile(tempPath, outputBaseFileName_fd_three);
    %outputFullFileName_folder_fd_num=fullfile(tempPath, outputBaseFileName_fd_num);
    
    %outputFullFileNameS_folder_temp_fd_three =fullfile(tempPath, outputBaseFileName_temp_fd_three);
    %outputFullFileName_folder_temp_fd_num =fullfile(tempPath, outputBaseFileName_temp_fd_num);
    
    
    %cd(temp)
    if exist(inputFullFileName_fd,'file')==2
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
        temp_fwd=fwd;
        temp_fwd(1:5,:)=[];
        temp_fd_mean=mean(temp_fwd);
        temp_fd_num=temp_fwd(temp_fwd(:,1)>0.20);
        temp_fd_percentage = (size(temp_fd_num,1)/size(temp_fwd,1));
        rounded_temp_fd_percentage=round(temp_fd_percentage,2);
        fd_mean=mean(fwd);
        fd_num=fwd(fwd(:,1)>0.20);
        fd_percentage= (size(fd_num,1)/size(fwd,1));
        rounded_fd_percentage=round(fd_percentage,2);
        if(rounded_temp_fd_percentage<=0.20)
            fid = fopen(fullfile(newFolder3, 'participantsListNew.txt'), 'a');
            if fid == -1
            error('Cannot open participants file.');
            end
            fprintf(fid, '%s\n',SubFold_fd(i).name);
            fclose(fid);
        end 
        
        if(rounded_fd_percentage<=0.20)
            fid = fopen(fullfile(newFolder2, 'participantsList.txt'), 'a');
            if fid == -1
            error('Cannot open participants file.');
            end
            fprintf(fid, '%s\n',SubFold_fd(i).name);
            fclose(fid);
        end     
        fd_three=[fd_mean,size(fd_num,1),size(fwd,1),rounded_fd_percentage];
        temp_fd_three=[temp_fd_mean,size(temp_fd_num,1),size(temp_fwd,1),rounded_temp_fd_percentage];
        %dlmwrite(outputFullFileNameS_folder_temp_fd_three,temp_fd_three)
        dlmwrite(outputFullFileName_temp_fd_three,temp_fd_three)
        dlmwrite(outputFullFileName_fd_three,fd_three)
        %dlmwrite(outputFullFileNameS_folder_fd_three,fd_three)
        dlmwrite(outputFullFileName_fd,fwd)
        %dlmwrite(outputFullFileNameS_folder_fd,fwd)
        
        dlmwrite(outputFullFileName_temp_fd,temp_fwd)
        %dlmwrite(outputFullFileNameS_folder_temp_fd,temp_fwd)
        
        
        %dlmwrite(outputFullFileName_folder_fd_num,fd_num)
        dlmwrite(outputFullFileName_fd_num,fd_num)
        %dlmwrite(outputFullFileName_folder_temp_fd_num,temp_fd_num)
        dlmwrite(outputFullFileName_temp_fd_num,temp_fd_num)
    else
    fid = fopen(fullfile(newFolder2, 'Log_Missing_data_for_subjects.txt'), 'a');
    if fid == -1
    error('Cannot open log file.');
    end
    fprintf(fid, '%s\n',SubFold_fd(i).name);
    fclose(fid);
    end
    
    
end
