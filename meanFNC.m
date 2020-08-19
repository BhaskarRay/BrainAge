group_ica_fmri=load('c_n15.mat');
meanFNC2=mean(group_ica_fmri.c_n,3);
figure; 
imagesc(meanFNC2);colorbar;
set(gca,'XTick',[1,6,8,14,23,29,43,54])
set(gca,'XTickLabel',{'SC','Ins','AU','SM','VI','CC','DMN','CB'})
set(gca,'YTick',[1,6,8,14,23,29,43,54])
set(gca,'YTickLabel',{'SC','Ins','AU','SM','VI','CC','DMN','CB'})