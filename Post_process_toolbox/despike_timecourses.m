function [TCnew, TCfit] = despike_timecourses(TC, plotit)
% DESPIKES THE DETRENDED TC [T x C]

%outputdir = '/export/mialab/hcp/dynamics';
%outname = 'TC_detrended_despiked.mat';
%inname = '/export/mialab/hcp/dynamics/TC_detrended.mat';

%addpath(genpath('/export/mialab/users/eallen/ALFF'))
%% load in the detrended/despiked timecourses
%load(inname); % loads TC [T x C]
if nargin < 2
    plotit = 0;
end


c1 = 2.5;
c2 = 3;


TCnew = zeros(size(TC));
TCfit = zeros(size(TC));
for kk = 1:size(TC,2),
    tc = TC(:,kk);
    tc = tc(:);
    
    %% first clean up the TCs a bit to determine where the outliers are
        p=3; % level of detrending
        r = size(TC,1);
        b = ((1 : r)' * ones (1, p + 1)) .^ (ones (r, 1) * (0 : p));  % build the regressors
        lestimates = robustfit(b(:,2:end), tc);
        yfit = b*lestimates;
        res = tc - yfit;    
 
    mad_res = median(abs(res - median(res))); % median absolute deviation of residuals
    sigma = mad_res* sqrt(pi/2);
    s = res/sigma;
    
    
    %% Here are the original points idenitified as outliers
    ind = find(abs(s) > c1);
    
    %% use a spline fit with few breakpoints o correct spikes that might also be end points
    % since the higher order spline fit doesn't deal well with those.
    if any(ismember(ind, [1, length(s)]))
        xaxis = setdiff(1:length(tc),[ind]);
        ytemp = tc(xaxis);
        bpfrac = 0.05;
        sporder = 3;
        pp0 = splinefit(xaxis,ytemp,floor(length(tc)*bpfrac),sporder);  % Piecewise quadratic
        y0 = ppval(pp0,1:length(tc));
        tc(ismember(ind, [1, length(s)])) = y0(ismember(ind, [1, length(s)]));
    end
    
    %% Method from AFNI that just lowers the level of the spike
    %     s_out = s;
    %     for uu = 1:length(ind)
    %         if ind(uu) == 1 || ind(uu) == length(s)
    %             s_out(ind(uu)) = sign(s(ind(uu)))*(c1+((c2-c1)*tanh((abs(s(ind(uu)))-c1)/(c2-c1))));
    %         else
    %             %do nothing
    %         end
    %     end
    %    tc = yfit + s_out*sigma;
    
    %% remove those indices that have already been dealt with
    ind = setdiff(ind, [1,length(s)]);
    
    
    %% use a spline fit to determine the new values
    xaxis = setdiff(1:length(tc),ind);
    ytemp = tc(xaxis);
    bpfrac = 0.5;
    sporder = 3;
    pp = splinefit(xaxis,ytemp,floor(length(tc)*bpfrac),sporder);  % Piecewise quadratic
    y1 = ppval(pp,1:length(tc));
    tcout = tc;
    tcout(ind) = y1(ind);
    %% Create a figure
    if plotit
    figure; plot(1:length(tc), TC(:,kk), 'k')
    hold on
    plot(ind, tc(ind), 'r*')
    plot(1:length(tc), tcout, 'g')
    hold on
    plot(1:length(tc), y1, ['r--'])
    end
    %% Store it
    TCnew(:,kk) = tcout; % just replace the "bad" points
    TCfit(:,kk) = y1; % fit from the spline
end
%disp(['finished subj: ' num2str(ii) 'in  time : ' num2str(toc) ' seconds' ])

%end






function [yfit] = getSplineFit(estimates,len,TR)

numP = floor(len/30);
t = 0:TR:(len-1)*TR;
t = t(:);
x0  = estimates;
yfit = x0(1)*t + x0(2)*t.^2;
for ii = 1:numP
    yfit = yfit + x0(2+ii) * sin(2*pi*ii*t/(len*TR)) + x0(2+ii) *cos(2*pi*ii*t/(len*TR));
end


function [yfit] = getQuadFit(estimates,len,TR)

numP = floor(len/30);
t = 0:TR:(len-1)*TR;
t = t(:);
x0  = estimates;
yfit = x0(1)*t.^2 + x0(2)*t + x0(3);
