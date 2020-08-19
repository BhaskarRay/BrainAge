% AAL atlas, average gray matter for each ROI
% data_smri: gray matter data, subject x voxel
% vind_mask: gray matter mask (i.e. voxel index) of the current dataset


%=== AAL atlas SPM8 version ==========================
load(fullfile(filepath_aal, 'ROI_MNI_V4_List.mat'));
aal_img = spm_read_vols(spm_vol(fullfile(filepath_aal, 'ROI_MNI_V4.nii')));
aal_img = aal_img(:);
clear vind_mask_aal data_smri_aal aal_roi_list;
for j = 1:length(ROI)
    aal_roi_list(j,:) = [ROI(j).Nom_L, {num2str(ROI(j).ID)}];
    vind_mask_aal{j} = find(aal_img(vind_mask) == ROI(j).ID);
    data_smri_aal(:,j) = nanmean(data_smri(:,vind_mask_aal{j}),2);
end
    
    

cd('/export/mialab/users/jchen/atlas/aal/aal_for_SPM8');
load aal_mask_121_145_121;
aal_label = [aal_label,cellstr(num2str(aal_id))];
open aal_label;
aal_mask = spm_read_vols(spm_vol('ROI_MNI_V4_121_145_121.nii'));
aal_mask = aal_mask(:);
aal_mask = aal_mask(sind_mask);

sind_aal_frontoparietal_l = [2101,2201,2601,6201,6211,6221];
sind_aal_frontoparietal_r = sind_aal_frontoparietal_l + 1;
sind_aal_frontoparietal = union(sind_aal_frontoparietal_l, sind_aal_frontoparietal_r);

sind_voxel_frontoparietal = [];
for j = sind_aal_frontoparietal
    sind = find(aal_mask == j);
    sind_voxel_frontoparietal = [sind_voxel_frontoparietal;sind];
end
size(sind_voxel_frontoparietal)


%=== pca =============================
sind_sub = [sind_cobre;sind_mcic;sind_nw;sind_jb];
icadata_smri = data_smrisc(sind_voxel_frontoparietal,sind_sub);
% icadata_smri = icadata_smri - repmat(mean(icadata_smri,2),1,size(icadata_smri,2));
icadata_smri = icadata_smri - repmat(mean(icadata_smri),size(icadata_smri,1),1);
[u1,s1,v1] = svd(icadata_smri,'econ');
plot(diag(s1),'.')