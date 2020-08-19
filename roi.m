load("C:/Users/bray14/Downloads/ROI_Approach/aal_for_SPM8/aal_for_SPM8/aal_mask_121_145_121.mat");
%clear data_smri_aal_test data_smri_aal_with_gray_matter_mask_test;
for j = 1:116
    %aal_roi_list(j,:) = [ROI(j).Nom_L, {num2str(ROI(j).ID)}];
    %vind_mask_aal{j} = find(aal_img(vind_mask) == ROI(j).ID);
    %data_smri_aal(:,j) = nanmean(data_smri(:,aal_mask{j}),2);
    %data_smri_aal(:,j) = nanmean(sMRItransdataTranspose(:,aal_mask{j}),2);
    %data_smri_aal_new(:,j) = nanmean(sMRIdata1(aal_mask{j},:),1);
    %data_smri_aal_with_gray_matter_mask(:,j) = nanmean(sMRIdata1(intersect(sind_mask,aal_mask{j}),:),1);
    %data_smri_aal_test(:,j) = nanmean(sMRIdata1QC(aal_mask{j},:),1);
    data_smri_aal_with_gray_matter_mask_test(:,j) = nanmean(sMRIdata1QC(intersect(gray_mask,aal_mask{j}),:),1);
    %full_data_smri_aal_with_gray_matter_mask_test(:,j) = nanmean(sMRIdata1(intersect(gray_mask,aal_mask{j}),:),1);
end

%csvwrite('C:/Users/bray14/Desktop/code_fd/temp_Final/RoiFeatures.csv',data_smri_aal);
%csvwrite('C:/Users/bray14/Desktop/code_fd/temp_Final/RoiFeatures2.csv',data_smri_aal_new);
%csvwrite('C:/Users/bray14/Desktop/code_fd/temp_Final/RoiFeatures_with_grayMask.csv',data_smri_aal_with_gray_matter_mask);
%save("data_smri_aal_new.mat","data_smri_aal_new");
%newFeatures = load('C:/Users/bray14/Downloads/new/sMRI data preproccessing and ICA code for Jessica/data_smri_aal_new.mat');

%roidatasetTEST = readtable('C:/Users/bray14/Desktop/code_fd/temp_Final/RoidataSetMedSIP_AGE.csv','TreatAsEmpty',{'.','NA','N/A'});


AAL_116_RoidataSet = data_smri_aal_with_gray_matter_mask_test;
csvwrite('C:/Users/bray14/Desktop/code_fd/temp_server/DatasetFinal/AAL_116_RoidataSet.csv',AAL_116_RoidataSet);
AAL_116_RoidataSetAge = [data_smri_aal_with_gray_matter_mask_test veriableOfInerest];
csvwrite('C:/Users/bray14/Desktop/code_fd/temp_server/DatasetFinal/AAL_116_RoidataSetAge.csv',AAL_116_RoidataSetAge);
AAL_116_cORRmAT_ROI = corr(AAL_116_RoidataSetAge);
plot(AAL_116_cORRmAT_ROI(:,117),'*');