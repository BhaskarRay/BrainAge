%load(fullfile(filepath_aal, 'ROI_MNI_V4_List.mat'));
aal_img_new_shen = spm_read_vols(spm_vol('C:/Users/bray14/OneDrive - Georgia State University/Fresh_ICA_run_100_components/Roi_ATLAS/shen_1mm_268_parcellation.nii/rshen_1mm_268_parcellation.nii'));
aal_img_new_shen = aal_img_new_shen(:);
clear gray_mask_aal_shen data_smri_aal_shen;
for j = 1:268
    %aal_roi_list(j,:) = [ROI(j).Nom_L, {num2str(ROI(j).ID)}];
    gray_mask_aal_shen{j} = find(aal_img_new_shen(gray_mask) == j);
    data_smri_aal_shen(:,j) = nanmean(sMRIdata1QC(gray_mask_aal_shen{j},:),1);
end

Shen_RoidataSet = data_smri_aal_shen;
csvwrite('C:/Users/bray14/Desktop/code_fd/temp_server/DatasetFinal/Shen_RoidataSet.csv',Shen_RoidataSet);
dlmwrite('C:/Users/bray14/Desktop/code_fd/temp_server/DatasetFinal/Shen_Roi_dataSetOrPresn.csv',Shen_RoidataSet,'delimiter',',','precision',100);
Shen_RoidataSetAge = [data_smri_aal_shen veriableOfInerest];
csvwrite('C:/Users/bray14/Desktop/code_fd/temp_server/DatasetFinal/Shen_RoidataSetAge.csv',Shen_RoidataSetAge);
Shen_cORRmAT_ROI = corr(Shen_RoidataSetAge);
plot(Shen_cORRmAT_ROI(:,269),'.');