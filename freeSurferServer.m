%clear;
yourpath_fd = '/data/mialab/UPENN/AUTO_ANALYSIS_BIDS/version2/derivatives/';
newFolder2 = '/data/mialab/users/bray14/';


%yourpath_fd = 'C:/Users/bray14/Desktop/freeSurfer2/';
%yourpath_fd = 'C:/Users/bray14/Desktop/freeSurfer2/version2/';
%newFolder2 = 'C:/Users/bray14/Desktop/freeSurfer2/';
% Get all the subfolders
ContentInFold_fd = dir(yourpath_fd);
SubFold_fd = ContentInFold_fd([ContentInFold_fd.isdir]); % keep only the directories
% Loop on each folder

%{

SUBJID ={};

%lh & rh subcortical region
eTIV_v={};

lh_lateral_ventricle_v ={};
lh_cerebellum_cortex_v ={};
lh_thalamus_proper_v ={};
lh_caudate_v ={};
lh_putamen_v ={};
lh_pallidum_v ={};
lh_hippocampus_v ={};
lh_amygdala_v ={};
lh_accumbens_area_v ={};

rh_lateral_ventricle_v ={};
rh_cerebellum_cortex_v ={};
rh_thalamus_proper_v ={};
rh_caudate_v ={};
rh_putamen_v ={};
rh_pallidum_v ={};
rh_hippocampus_v ={};
rh_amygdala_v ={};
rh_accumbens_area_v ={};



%lh_volume
lh_bankssts_v = {};
lh_caudalanteriorcingulate_v = {};
lh_caudalmiddlefrontal_v = {};
lh_cuneus_v = {};
lh_entorhinal_v = {};
lh_fusiform_v = {};
lh_inferiorparietal_v = {};
lh_inferiortemporal_v = {};
lh_isthmuscingulate_v = {};
lh_lateraloccipital_v = {};
lh_lateralorbitofrontal_v = {};
lh_lingual_v = {};
lh_medialorbitofrontal_v = {};
lh_middletemporal_v = {};
lh_parahippocampal_v = {};
lh_paracentral_v = {};
lh_parsopercularis_v = {};
lh_parsorbitalis_v = {};
lh_parstriangularis_v = {};
lh_pericalcarine_v = {};
lh_postcentral_v = {};
lh_posteriorcingulate_v = {};
lh_precentral_v = {};
lh_precuneus_v = {};
lh_rostralanteriorcingulate_v = {};
lh_rostralmiddlefrontal_v = {};
lh_superiorfrontal_v = {};
lh_superiorparietal_v = {};
lh_superiortemporal_v = {};
lh_supramarginal_v = {};
lh_frontalpole_v = {};
lh_temporalpole_v = {};
lh_transversetemporal_v = {};
lh_insula_v = {};

%rh_volume
rh_bankssts_v = {};
rh_caudalanteriorcingulate_v = {};
rh_caudalmiddlefrontal_v = {};
rh_cuneus_v = {};
rh_entorhinal_v = {};
rh_fusiform_v = {};
rh_inferiorparietal_v = {};
rh_inferiortemporal_v = {};
rh_isthmuscingulate_v = {};
rh_lateraloccipital_v = {};
rh_lateralorbitofrontal_v = {};
rh_lingual_v = {};
rh_medialorbitofrontal_v = {};
rh_middletemporal_v = {};
rh_parahippocampal_v = {};
rh_paracentral_v = {};
rh_parsopercularis_v = {};
rh_parsorbitalis_v = {};
rh_parstriangularis_v = {};
rh_pericalcarine_v = {};
rh_postcentral_v = {};
rh_posteriorcingulate_v = {};
rh_precentral_v = {};
rh_precuneus_v = {};
rh_rostralanteriorcingulate_v = {};
rh_rostralmiddlefrontal_v = {};
rh_superiorfrontal_v = {};
rh_superiorparietal_v = {};
rh_superiortemporal_v = {};
rh_supramarginal_v = {};
rh_frontalpole_v = {};
rh_temporalpole_v = {};
rh_transversetemporal_v = {};
rh_insula_v = {};

%lh_thickness
lh_bankssts_t = {};
lh_caudalanteriorcingulate_t = {};
lh_caudalmiddlefrontal_t = {};
lh_cuneus_t = {};
lh_entorhinal_t = {};
lh_fusiform_t = {};
lh_inferiorparietal_t = {};
lh_inferiortemporal_t = {};
lh_isthmuscingulate_t = {};
lh_lateraloccipital_t = {};
lh_lateralorbitofrontal_t = {};
lh_lingual_t = {};
lh_medialorbitofrontal_t = {};
lh_middletemporal_t = {};
lh_parahippocampal_t = {};
lh_paracentral_t = {};
lh_parsopercularis_t = {};
lh_parsorbitalis_t = {};
lh_parstriangularis_t = {};
lh_pericalcarine_t = {};
lh_postcentral_t = {};
lh_posteriorcingulate_t = {};
lh_precentral_t = {};
lh_precuneus_t = {};
lh_rostralanteriorcingulate_t = {};
lh_rostralmiddlefrontal_t = {};
lh_superiorfrontal_t = {};
lh_superiorparietal_t = {};
lh_superiortemporal_t = {};
lh_supramarginal_t = {};
lh_frontalpole_t = {};
lh_temporalpole_t = {};
lh_transversetemporal_t = {};
lh_insula_t = {};
lh_MeanThickness_t = {};

%rh_thickness
rh_bankssts_t = {};
rh_caudalanteriorcingulate_t = {};
rh_caudalmiddlefrontal_t = {};
rh_cuneus_t = {};
rh_entorhinal_t = {};
rh_fusiform_t = {};
rh_inferiorparietal_t = {};
rh_inferiortemporal_t = {};
rh_isthmuscingulate_t = {};
rh_lateraloccipital_t = {};
rh_lateralorbitofrontal_t = {};
rh_lingual_t = {};
rh_medialorbitofrontal_t = {};
rh_middletemporal_t = {};
rh_parahippocampal_t = {};
rh_paracentral_t = {};
rh_parsopercularis_t = {};
rh_parsorbitalis_t = {};
rh_parstriangularis_t = {};
rh_pericalcarine_t = {};
rh_postcentral_t = {};
rh_posteriorcingulate_t = {};
rh_precentral_t = {};
rh_precuneus_t = {};
rh_rostralanteriorcingulate_t = {};
rh_rostralmiddlefrontal_t = {};
rh_superiorfrontal_t = {};
rh_superiorparietal_t = {};
rh_superiortemporal_t = {};
rh_supramarginal_t = {};
rh_frontalpole_t = {};
rh_temporalpole_t = {};
rh_transversetemporal_t = {};
rh_insula_t = {};
rh_MeanThickness_t = {};


%}

for i = 3:length(SubFold_fd)
    if SubFold_fd(i).name ~= "sub-609712666246"
        tempPath = fullfile(yourpath_fd,SubFold_fd(i).name,'analysis','fs_53','stats','//'); 
        %txtfile_aseg = dir([tempPath,'aseg.stats']);
        filename = strcat(tempPath,'aseg.stats');

        if exist(filename,'file')==2
            fid=fopen(filename);
            SUBJID(end+1,:)=cellstr(SubFold_fd(i).name(5:16));
            linenum = 34;
            % use '%s' if you want to read in the entire line or use '%f' if you want to read only the first numeric value
            eTIV = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(eTIV{1});
            part_eTIV_Data = split(str,",");
            %header_eTIV = part_eTIV_Data(3);
            data_eTIV = part_eTIV_Data(4);
            eTIV_v(end+1,:)=data_eTIV;




            %line 80
            frewind(fid);
            linenum = 80;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            lh_lateral_ventricle_v(end+1,:)=data;

            %line 83
            frewind(fid);
            linenum = 83;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5);  
            data = part_temp_Data(4);
            lh_cerebellum_cortex_v(end+1,:)=data;

            %line 84
            frewind(fid);
            linenum = 84;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5);
            data = part_temp_Data(4);
            lh_thalamus_proper_v(end+1,:)=data;

            %line 85
            frewind(fid);
            linenum = 85;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            lh_caudate_v(end+1,:)=data;

            %line 86
            frewind(fid);
            linenum = 86;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5);  
            data = part_temp_Data(4);
            lh_putamen_v(end+1,:)=data;

            %line 87
            frewind(fid);
            linenum = 87;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            header = part_temp_Data(5);  
            data = part_temp_Data(4);
            lh_pallidum_v(end+1,:)=data;

            %line 91
            frewind(fid);
            linenum = 91;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            lh_hippocampus_v(end+1,:)=data;


            %line 92
            frewind(fid);
            linenum = 92;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            lh_amygdala_v(end+1,:)=data;


            %line 94
            frewind(fid);
            linenum = 94;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            lh_accumbens_area_v(end+1,:)=data;

            %line 98
            frewind(fid);
            linenum = 98;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            rh_lateral_ventricle_v(end+1,:)=data;


            %line 101
            frewind(fid);
            linenum = 101;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5);  
            data = part_temp_Data(4);
            rh_cerebellum_cortex_v(end+1,:)=data;

            %line 102
            frewind(fid);
            linenum = 102;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            rh_thalamus_proper_v(end+1,:)=data;


            %line 103
            frewind(fid);
            linenum = 103;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5);  
            data = part_temp_Data(4);
            rh_caudate_v(end+1,:)=data;

            %line 104
            frewind(fid);
            linenum = 104;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            rh_putamen_v(end+1,:)=data;

            %line 105
            frewind(fid);
            linenum = 105;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            rh_pallidum_v(end+1,:)=data;

            %line 106
            frewind(fid);
            linenum = 106;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5);  
            data = part_temp_Data(4);
            rh_hippocampus_v(end+1,:)=data;

            %line 107
            frewind(fid);
            linenum = 107;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5); 
            data = part_temp_Data(4);
            rh_amygdala_v(end+1,:)=data;


            %line 108
            frewind(fid);
            linenum = 108;
            temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
            str = char(temp{1});
            part_temp_Data = strsplit(str);
            %header = part_temp_Data(5);  
            data = part_temp_Data(4);
            rh_accumbens_area_v(end+1,:)=data;

            fclose(fid);

        else
        fid = fopen(fullfile(newFolder2, 'logFile_aseg.txt'), 'a');
        if fid == -1
        error('Cannot open log file.');
        end
        fprintf(fid, '%s\n',SubFold_fd(i).name);
        fclose(fid); 
        end






        lh_volume_name = strcat(SubFold_fd(i).name,'_lh_aparc_volume.txt');
        %txtfile_lh_volume = dir([tempPath,lh_volume_name]);
        filename = strcat(tempPath,lh_volume_name);

        if exist(filename,'file')==2
            fid=fopen(filename);

            for j=2:35
                frewind(fid);
                %linenum = 2;
                linenum = j;
                temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
                str = char(temp{1});
                part_temp_Data = strsplit(str);
                header1 = part_temp_Data(1);
                header1_data = part_temp_Data(2);

                if j == 2
                lh_bankssts_v(end+1,:)= header1_data;
                end

                if j == 3
                lh_caudalanteriorcingulate_v(end+1,:)= header1_data;
                end

                if j == 4
                lh_caudalmiddlefrontal_v(end+1,:)= header1_data;
                end

                if j == 5
                lh_cuneus_v(end+1,:)= header1_data;
                end

                if j == 6
                lh_entorhinal_v(end+1,:)= header1_data;
                end

                if j == 7
                lh_fusiform_v(end+1,:)= header1_data;
                end

                if j == 8
                lh_inferiorparietal_v(end+1,:)= header1_data;
                end

                if j == 9
                lh_inferiortemporal_v(end+1,:)= header1_data;
                end

                if j == 10
                lh_isthmuscingulate_v(end+1,:)= header1_data;
                end

                if j == 11
                lh_lateraloccipital_v(end+1,:)= header1_data;
                end

                if j == 12
                lh_lateralorbitofrontal_v(end+1,:)= header1_data;
                end

                if j == 13
                lh_lingual_v(end+1,:)= header1_data;
                end

                if j == 14
                lh_medialorbitofrontal_v(end+1,:)= header1_data;
                end

                if j == 15
                lh_middletemporal_v(end+1,:)= header1_data;
                end

                if j == 16
                lh_parahippocampal_v(end+1,:)= header1_data;
                end

                if j == 17
                lh_paracentral_v(end+1,:)= header1_data;
                end

                if j == 18
                lh_parsopercularis_v(end+1,:)= header1_data;
                end

                if j == 19
                lh_parsorbitalis_v(end+1,:)= header1_data;
                end

                if j == 20
                lh_parstriangularis_v(end+1,:)= header1_data;
                end

                if j == 21
                lh_pericalcarine_v(end+1,:)= header1_data;
                end

                if j == 22
                lh_postcentral_v(end+1,:)= header1_data;
                end

                if j == 23
                lh_posteriorcingulate_v(end+1,:)= header1_data;
                end

                if j == 24
                lh_precentral_v(end+1,:)= header1_data;
                end

                if j == 25
                lh_precuneus_v(end+1,:)= header1_data;
                end

                if j == 26
                lh_rostralanteriorcingulate_v(end+1,:)= header1_data;
                end

                if j == 27
                lh_rostralmiddlefrontal_v(end+1,:)= header1_data;
                end

                if j == 28
                lh_superiorfrontal_v(end+1,:)= header1_data;
                end

                if j == 29
                lh_superiorparietal_v(end+1,:)= header1_data;
                end

                if j == 30
                lh_superiortemporal_v(end+1,:)= header1_data;
                end

                if j == 31
                lh_supramarginal_v(end+1,:)= header1_data;
                end

                if j == 32
                lh_frontalpole_v(end+1,:)= header1_data;
                end

                if j == 33
                lh_temporalpole_v(end+1,:)= header1_data;
                end

                if j == 34
                lh_transversetemporal_v(end+1,:)= header1_data;
                end

                if j == 35
                lh_insula_v(end+1,:)= header1_data;
                end




        end

        fclose(fid);

        else
        fid = fopen(fullfile(newFolder2, 'logFile_lh_aparc_volume.txt'), 'a');
        if fid == -1
        error('Cannot open log file.');
        end
        fprintf(fid, '%s\n',SubFold_fd(i).name);
        fclose(fid);    
        end


        rh_volume_name = strcat(SubFold_fd(i).name,'_rh_aparc_volume.txt');
        %txtfile_rh_volume = dir([tempPath,rh_volume_name]);
        filename = strcat(tempPath,lh_volume_name);
        fid=fopen(filename);

        if exist(filename,'file')==2
            for k=2:35
                frewind(fid);
                %linenum = 2;
                linenum = k;
                temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
                str = char(temp{1});
                part_temp_Data = strsplit(str);
                header1 = part_temp_Data(1);
                header1_data = part_temp_Data(2);

                if k == 2
                rh_bankssts_v(end+1,:)= header1_data;
                end

                if k == 3
                rh_caudalanteriorcingulate_v(end+1,:)= header1_data;
                end

                if k == 4
                rh_caudalmiddlefrontal_v(end+1,:)= header1_data;
                end

                if k == 5
                rh_cuneus_v(end+1,:)= header1_data;
                end

                if k == 6
                rh_entorhinal_v(end+1,:)= header1_data;
                end

                if k == 7
                rh_fusiform_v(end+1,:)= header1_data;
                end

                if k == 8
                rh_inferiorparietal_v(end+1,:)= header1_data;
                end

                if k == 9
                rh_inferiortemporal_v(end+1,:)= header1_data;
                end

                if k == 10
                rh_isthmuscingulate_v(end+1,:)= header1_data;
                end

                if k == 11
                rh_lateraloccipital_v(end+1,:)= header1_data;
                end

                if k == 12
                rh_lateralorbitofrontal_v(end+1,:)= header1_data;
                end

                if k == 13
                rh_lingual_v(end+1,:)= header1_data;
                end

                if k == 14
                rh_medialorbitofrontal_v(end+1,:)= header1_data;
                end

                if k == 15
                rh_middletemporal_v(end+1,:)= header1_data;
                end

                if k == 16
                rh_parahippocampal_v(end+1,:)= header1_data;
                end

                if k == 17
                rh_paracentral_v(end+1,:)= header1_data;
                end

                if k == 18
                rh_parsopercularis_v(end+1,:)= header1_data;
                end

                if k == 19
                rh_parsorbitalis_v(end+1,:)= header1_data;
                end

                if k == 20
                rh_parstriangularis_v(end+1,:)= header1_data;
                end

                if k == 21
                rh_pericalcarine_v(end+1,:)= header1_data;
                end

                if k == 22
                rh_postcentral_v(end+1,:)= header1_data;
                end

                if k == 23
                rh_posteriorcingulate_v(end+1,:)= header1_data;
                end

                if k == 24
                rh_precentral_v(end+1,:)= header1_data;
                end

                if k == 25
                rh_precuneus_v(end+1,:)= header1_data;
                end

                if k == 26
                rh_rostralanteriorcingulate_v(end+1,:)= header1_data;
                end

                if k == 27
                rh_rostralmiddlefrontal_v(end+1,:)= header1_data;
                end

                if k == 28
                rh_superiorfrontal_v(end+1,:)= header1_data;
                end

                if k == 29
                rh_superiorparietal_v(end+1,:)= header1_data;
                end

                if k == 30
                rh_superiortemporal_v(end+1,:)= header1_data;
                end

                if k == 31
                rh_supramarginal_v(end+1,:)= header1_data;
                end

                if k == 32
                rh_frontalpole_v(end+1,:)= header1_data;
                end

                if k == 33
                rh_temporalpole_v(end+1,:)= header1_data;
                end

                if k == 34
                rh_transversetemporal_v(end+1,:)= header1_data;
                end

                if k == 35
                rh_insula_v(end+1,:)= header1_data;
                end


            end

        fclose(fid);
        else   
        fid = fopen(fullfile(newFolder2, 'logFile_rh_aparc_volume.txt'), 'a');
        if fid == -1
        error('Cannot open log file.');
        end
        fprintf(fid, '%s\n',SubFold_fd(i).name);
        fclose(fid);  
        end

        lh_thickness_name = strcat(SubFold_fd(i).name,'_lh_aparc_thickness.txt');
        %txtfile_rh_volume = dir([tempPath,rh_volume_name]);
        filename = strcat(tempPath,lh_thickness_name);
        fid=fopen(filename);
        if exist(filename,'file')==2
            for k=2:36
                frewind(fid);
                %linenum = 2;
                linenum = k;
                temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
                str = char(temp{1});
                part_temp_Data = strsplit(str);
                header1 = part_temp_Data(1);
                header1_data = part_temp_Data(2);

                if k == 2
                lh_bankssts_t(end+1,:)= header1_data;
                end

                if k == 3
                lh_caudalanteriorcingulate_t(end+1,:)= header1_data;
                end

                if k == 4
                lh_caudalmiddlefrontal_t(end+1,:)= header1_data;
                end

                if k == 5
                lh_cuneus_t(end+1,:)= header1_data;
                end

                if k == 6
                lh_entorhinal_t(end+1,:)= header1_data;
                end

                if k == 7
                lh_fusiform_t(end+1,:)= header1_data;
                end

                if k == 8
                lh_inferiorparietal_t(end+1,:)= header1_data;
                end

                if k == 9
                lh_inferiortemporal_t(end+1,:)= header1_data;
                end

                if k == 10
                lh_isthmuscingulate_t(end+1,:)= header1_data;
                end

                if k == 11
                lh_lateraloccipital_t(end+1,:)= header1_data;
                end

                if k == 12
                lh_lateralorbitofrontal_t(end+1,:)= header1_data;
                end

                if k == 13
                lh_lingual_t(end+1,:)= header1_data;
                end

                if k == 14
                lh_medialorbitofrontal_t(end+1,:)= header1_data;
                end

                if k == 15
                lh_middletemporal_t(end+1,:)= header1_data;
                end

                if k == 16
                lh_parahippocampal_t(end+1,:)= header1_data;
                end

                if k == 17
                lh_paracentral_t(end+1,:)= header1_data;
                end

                if k == 18
                lh_parsopercularis_t(end+1,:)= header1_data;
                end

                if k == 19
                lh_parsorbitalis_t(end+1,:)= header1_data;
                end

                if k == 20
                lh_parstriangularis_t(end+1,:)= header1_data;
                end

                if k == 21
                lh_pericalcarine_t(end+1,:)= header1_data;
                end

                if k == 22
                lh_postcentral_t(end+1,:)= header1_data;
                end

                if k == 23
                lh_posteriorcingulate_t(end+1,:)= header1_data;
                end

                if k == 24
                lh_precentral_t(end+1,:)= header1_data;
                end

                if k == 25
                lh_precuneus_t(end+1,:)= header1_data;
                end

                if k == 26
                lh_rostralanteriorcingulate_t(end+1,:)= header1_data;
                end

                if k == 27
                lh_rostralmiddlefrontal_t(end+1,:)= header1_data;
                end

                if k == 28
                lh_superiorfrontal_t(end+1,:)= header1_data;
                end

                if k == 29
                lh_superiorparietal_t(end+1,:)= header1_data;
                end

                if k == 30
                lh_superiortemporal_t(end+1,:)= header1_data;
                end

                if k == 31
                lh_supramarginal_t(end+1,:)= header1_data;
                end

                if k == 32
                lh_frontalpole_t(end+1,:)= header1_data;
                end

                if k == 33
                lh_temporalpole_t(end+1,:)= header1_data;
                end

                if k == 34
                lh_transversetemporal_t(end+1,:)= header1_data;
                end

                if k == 35
                lh_insula_t(end+1,:)= header1_data;
                end

                if k == 36
                lh_MeanThickness_t(end+1,:)= header1_data;
                end


            end

        fclose(fid);
        else   
        fid = fopen(fullfile(newFolder2, 'logFile_lh_aparc_thickness.txt'), 'a');
        if fid == -1
        error('Cannot open log file.');
        end
        fprintf(fid, '%s\n',SubFold_fd(i).name);
        fclose(fid);  
        end


        rh_thickness_name = strcat(SubFold_fd(i).name,'_rh_aparc_thickness.txt');
        %txtfile_rh_volume = dir([tempPath,rh_volume_name]);
        filename = strcat(tempPath,rh_thickness_name);
        fid=fopen(filename);
        if exist(filename,'file')==2
            for k=2:36
                frewind(fid);
                %linenum = 2;
                linenum = k;
                temp = textscan(fid,'%s',1,'delimiter','\n', 'headerlines',linenum-1);
                str = char(temp{1});
                part_temp_Data = strsplit(str);
                header1 = part_temp_Data(1);
                header1_data = part_temp_Data(2);

                if k == 2
                rh_bankssts_t(end+1,:)= header1_data;
                end

                if k == 3
                rh_caudalanteriorcingulate_t(end+1,:)= header1_data;
                end

                if k == 4
                rh_caudalmiddlefrontal_t(end+1,:)= header1_data;
                end

                if k == 5
                rh_cuneus_t(end+1,:)= header1_data;
                end

                if k == 6
                rh_entorhinal_t(end+1,:)= header1_data;
                end

                if k == 7
                rh_fusiform_t(end+1,:)= header1_data;
                end

                if k == 8
                rh_inferiorparietal_t(end+1,:)= header1_data;
                end

                if k == 9
                rh_inferiortemporal_t(end+1,:)= header1_data;
                end

                if k == 10
                rh_isthmuscingulate_t(end+1,:)= header1_data;
                end

                if k == 11
                rh_lateraloccipital_t(end+1,:)= header1_data;
                end

                if k == 12
                rh_lateralorbitofrontal_t(end+1,:)= header1_data;
                end

                if k == 13
                rh_lingual_t(end+1,:)= header1_data;
                end

                if k == 14
                rh_medialorbitofrontal_t(end+1,:)= header1_data;
                end

                if k == 15
                rh_middletemporal_t(end+1,:)= header1_data;
                end

                if k == 16
                rh_parahippocampal_t(end+1,:)= header1_data;
                end

                if k == 17
                rh_paracentral_t(end+1,:)= header1_data;
                end

                if k == 18
                rh_parsopercularis_t(end+1,:)= header1_data;
                end

                if k == 19
                rh_parsorbitalis_t(end+1,:)= header1_data;
                end

                if k == 20
                rh_parstriangularis_t(end+1,:)= header1_data;
                end

                if k == 21
                rh_pericalcarine_t(end+1,:)= header1_data;
                end

                if k == 22
                rh_postcentral_t(end+1,:)= header1_data;
                end

                if k == 23
                rh_posteriorcingulate_t(end+1,:)= header1_data;
                end

                if k == 24
                rh_precentral_t(end+1,:)= header1_data;
                end

                if k == 25
                rh_precuneus_t(end+1,:)= header1_data;
                end

                if k == 26
                rh_rostralanteriorcingulate_t(end+1,:)= header1_data;
                end

                if k == 27
                rh_rostralmiddlefrontal_t(end+1,:)= header1_data;
                end

                if k == 28
                rh_superiorfrontal_t(end+1,:)= header1_data;
                end

                if k == 29
                rh_superiorparietal_t(end+1,:)= header1_data;
                end

                if k == 30
                rh_superiortemporal_t(end+1,:)= header1_data;
                end

                if k == 31
                rh_supramarginal_t(end+1,:)= header1_data;
                end

                if k == 32
                rh_frontalpole_t(end+1,:)= header1_data;
                end

                if k == 33
                rh_temporalpole_t(end+1,:)= header1_data;
                end

                if k == 34
                rh_transversetemporal_t(end+1,:)= header1_data;
                end

                if k == 35
                rh_insula_t(end+1,:)= header1_data;
                end

                if k == 36
                rh_MeanThickness_t(end+1,:)= header1_data;
                end


            end

        fclose(fid);
        else   
        fid = fopen(fullfile(newFolder2, 'logFile_rh_aparc_thickness.txt'), 'a');
        if fid == -1
        error('Cannot open log file.');
        end
        fprintf(fid, '%s\n',SubFold_fd(i).name);
        fclose(fid);  
        end
    
    else
        fid = fopen(fullfile(newFolder2, 'logAvoid.txt'), 'a');
        if fid == -1
        error('Cannot open log file.');
        end
        fprintf(fid, '%s\n',SubFold_fd(i).name);
        fclose(fid); 
        
    end
    
    
    
end

freeSurferDataset = table(SUBJID,eTIV_v,lh_lateral_ventricle_v,lh_cerebellum_cortex_v,lh_thalamus_proper_v,lh_caudate_v,lh_putamen_v,lh_pallidum_v,lh_hippocampus_v,lh_amygdala_v,lh_accumbens_area_v,rh_lateral_ventricle_v,rh_cerebellum_cortex_v,rh_thalamus_proper_v,rh_caudate_v,rh_putamen_v,rh_pallidum_v,rh_hippocampus_v,rh_amygdala_v,rh_accumbens_area_v,lh_bankssts_v,lh_caudalanteriorcingulate_v,lh_caudalmiddlefrontal_v,lh_cuneus_v,lh_entorhinal_v,lh_fusiform_v,lh_inferiorparietal_v,lh_inferiortemporal_v,lh_isthmuscingulate_v,lh_lateraloccipital_v,lh_lateralorbitofrontal_v,lh_lingual_v,lh_medialorbitofrontal_v,lh_parahippocampal_v,lh_parsopercularis_v,lh_parstriangularis_v,lh_pericalcarine_v,lh_postcentral_v,lh_posteriorcingulate_v,lh_precentral_v,lh_precuneus_v,lh_rostralanteriorcingulate_v,lh_rostralmiddlefrontal_v,lh_superiorfrontal_v,lh_superiorparietal_v,lh_superiortemporal_v,lh_supramarginal_v,lh_temporalpole_v,lh_transversetemporal_v,lh_insula_v,rh_bankssts_v,rh_caudalanteriorcingulate_v,rh_caudalmiddlefrontal_v,rh_cuneus_v,rh_entorhinal_v,rh_fusiform_v,rh_inferiorparietal_v ,rh_inferiortemporal_v,rh_isthmuscingulate_v,rh_lateraloccipital_v,rh_lateralorbitofrontal_v,rh_lingual_v,rh_medialorbitofrontal_v,rh_middletemporal_v,rh_paracentral_v,rh_parsopercularis_v,rh_parsorbitalis_v,rh_parstriangularis_v,rh_pericalcarine_v,rh_postcentral_v,rh_posteriorcingulate_v,rh_precentral_v,rh_precuneus_v,rh_rostralanteriorcingulate_v,rh_rostralmiddlefrontal_v,rh_superiorfrontal_v,rh_superiorparietal_v,rh_superiortemporal_v,rh_supramarginal_v,rh_frontalpole_v,rh_temporalpole_v,rh_transversetemporal_v,rh_insula_v,lh_bankssts_t,lh_caudalanteriorcingulate_t,lh_caudalmiddlefrontal_t,lh_cuneus_t,lh_entorhinal_t,lh_fusiform_t,lh_inferiorparietal_t,lh_inferiortemporal_t,lh_isthmuscingulate_t,lh_lateraloccipital_t,lh_lateralorbitofrontal_t,lh_lingual_t,lh_medialorbitofrontal_t,lh_middletemporal_t,lh_parahippocampal_t,lh_paracentral_t,lh_parsopercularis_t,lh_parsorbitalis_t,lh_parstriangularis_t,lh_pericalcarine_t,lh_postcentral_t,lh_posteriorcingulate_t,lh_precentral_t,lh_precuneus_t,lh_rostralanteriorcingulate_t,lh_rostralmiddlefrontal_t,lh_superiorfrontal_t,lh_superiorparietal_t,lh_superiortemporal_t,lh_supramarginal_t,lh_frontalpole_t,lh_temporalpole_t,lh_transversetemporal_t,lh_insula_t,lh_MeanThickness_t,rh_bankssts_t,rh_caudalanteriorcingulate_t,rh_caudalmiddlefrontal_t,rh_cuneus_t,rh_entorhinal_t,rh_fusiform_t,rh_inferiorparietal_t,rh_inferiortemporal_t,rh_isthmuscingulate_t,rh_lateraloccipital_t,rh_lateralorbitofrontal_t,rh_lingual_t,rh_medialorbitofrontal_t,rh_middletemporal_t,rh_parahippocampal_t,rh_paracentral_t,rh_parsopercularis_t,rh_parsorbitalis_t,rh_parstriangularis_t,rh_pericalcarine_t,rh_postcentral_t,rh_posteriorcingulate_t,rh_precentral_t,rh_precuneus_t,rh_rostralanteriorcingulate_t,rh_rostralmiddlefrontal_t,rh_superiorfrontal_t,rh_superiorparietal_t,rh_superiortemporal_t,rh_supramarginal_t,rh_frontalpole_t,rh_temporalpole_t,rh_transversetemporal_t,rh_insula_t,rh_MeanThickness_t);
freeSurferDataset_new = freeSurferDataset{:,:};




