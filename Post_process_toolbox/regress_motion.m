function Y = regress_motion(TC, mp, p)

%% Build the detrending regressors
% p =  level of detrending
% 0 = mean, 1 = linear, 2 = quadratic, 3 = cubic
r = size(TC,1);
b = ((1 : r)' * ones (1, p + 1)) .^ (ones (r, 1) * (0 : p));  % build the regressors
b = zscore(b(:,2:end));
b = [ones(r,1), b];

%% Build the motion parameters with derivatives
mpX = [mp, [zeros(1,size(mp,2)); diff(mp)] ];
X = [b, zscore(mpX)];

Y = TC-X*(X\TC); % remove the best fit -- could replace this with robust fitting...