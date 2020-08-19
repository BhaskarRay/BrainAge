%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Postprocessing before calculation of FC or dFC
%%%% Zening Fu
%%%% 11.20.2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% input
% TC : timecourse of the IC, with dimension as timepoints * IC numbers
% TR
% Head_motion: with dimension as timepoints * 6
% Highcutoff: high cutoff frequency
% Lowcutoff: low cutoff frequency
%%%% Output
% Out_TC : output timecourse of the IC, with dimension as timepoints * IC

function [ Out_TC ] = Post_process_TC(TC, Head_motion, TR, Highcutoff, Lowcutoff, remove_head_opt, detrend_opt, despike_opt, filter_opt)
%1. Zscore the time courses
TC_z = zeros(size(TC));
for i = 1:size(TC, 2)
    TC_z(:,i) = zscore(TC(:,i));
%     TC_z(:,i) = (TC(:,i));
end

if remove_head_opt == 1
    %2. residualize with respect to motion/motion derivatives
    TC_z = regress_motion(TC_z, Head_motion, 3);
end

%3. detrending
if detrend_opt == 1
    TC_z = detrend(TC_z);
end

%4.remove large spikes from the TCs
if despike_opt == 1
    TC_z = despike_timecourses(TC_z, 0);
end

%5.filter
%filter define
if filter_opt
    NyqF = (1/TR)/2; %%Nyquist frequency
    wn   = [Lowcutoff/NyqF Highcutoff/NyqF];
    nf   = 5;
    [b1 a1] = butter(nf,wn);
    % filter and zscore
    TC_z  = zscore(filtfilt(b1,a1,TC_z));
end

%% save output
Out_TC = TC_z;

end



