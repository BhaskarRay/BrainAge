clc;close all; clear all;

addpath('./Post_process_toolbox');
%%%%1: Post processing (despike,detrend,filter, and motion) to generate the time course for each subject
TR = 3;%%modify to your TR
subj_num = 1113;%%modify the subject number
TC_filename=textread('./TC_filename.txt','%s');%%list of the full path of the timecourse file for each subject(timecourse files: in your GICA output folder, look for files end with '_ica_c1-1.mat' to '_ica_c1-626.mat')
MP_filename=textread('./MP_filename.txt','%s');%%list of the full path of the motion parameter file for each subject(motion parameter files: In fMRI autoanalysis group's preprocessing folder, look for files with name containing 'rp_sub-xxx.txt') 
TC15_str=cell(1,subj_num);
for i=1:size(TC_filename,1)
    disp(['processing subject ' num2str(i)])
    tmp_d = load(TC_filename{i});
    tmp_MP = textread(MP_filename{i});   
    TC15_str{i}= Post_process_TC(tmp_d.tc,tmp_MP(5:end,:),TR,.15,.01,1,1,1,1);%%%removed the first 4 time course
end
save('TC15_str.mat','TC15_str')


%%%%%%2: Generate FNC for 56 good components for each subject.
%ic_56: indices for selected components 
%ic_56=[2 3 4 5 7 8 10 11 13 14 16 17 18 19 23 25 27 29 31 34 35 41 45 46 48 49 51 52 53 54 55 61 62 64 65 67 68 69 71 78 80 82 84 85 86 87 88 89 90 93 94 95 96 97 98 20];
%ic_56=[12 3 4 7 6 9 11 13 15 18 20 19 24 5 17 87 14 23 33 37 45 96 61 41 55 28 75 89 51 29 59 50 44 78 70 64 68 74 66 73 76 53 81 85 47 67 100 57 97 79 92 88 83 80 82 8];
ic_56=[9 24 23 14 15 75 55 18 8 37 81 82 45 12 3 4 7 61 29 59 97 83 11 20 70 68 74 66 41 44 78 89 64 53 76 85 47 79 100 57 92 80 96 73 51 88 67 87 13 33 19 17 28 6 5 50];
c_n=zeros(length(ic_56),length(ic_56),subj_num);
diag1=diag(ones(1,length(ic_56)),0);
for i=1:subj_num
    a=TC15_str{i}(:,ic_56);
    c_n(:,:,i)=corr(a)-diag1;
end
save('c_n15.mat','c_n')

%%%plot mean FNC across subjects
meanFNC=mean(c_n,3);
figure; 
imagesc(meanFNC);colorbar;
set(gca,'XTick',[1,6,8,14,23,29,43,54])
set(gca,'XTickLabel',{'SC','Ins','AU','SM','VI','CC','DMN','CB'})
set(gca,'YTick',[1,6,8,14,23,29,43,54])
set(gca,'YTickLabel',{'SC','Ins','AU','SM','VI','CC','DMN','CB'})

