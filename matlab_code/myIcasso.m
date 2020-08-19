function [A, W, icasig, iq, metric_Q] = myIcasso(data, numOfIC, num_ica_runs, algorithm)
% data - pairs x (subjects x windows)
% numOfIC - Number of components
% num_ica_runs - Number of times to run ica
% algorithm - 1 - Infomax, 2 - Fast ica, 
%icatb_icaAlgorithm;

minClusterSize = ceil(0.8*num_ica_runs);
maxClusterSize = num_ica_runs;

%numOfIC = size(data, 1);

%%%%% Calculate PCA and Whitening matrix %%%%%
[V, Lambda] = icatb_v_pca(data, 1, numOfIC, 0, 'transpose', 'yes');

% Whiten matrix
[w, White, deWhite] = icatb_v_whiten(data, V, Lambda, 'transpose');

sR = icatb_icassoEst('randinit', data, num_ica_runs, 'numOfPC', numOfIC, 'algoIndex', algorithm, ...
    'dewhiteM', deWhite, 'whiteM', White, 'whitesig', w, 'icaOptions', {});

sR = icassoExp(sR);

% 
 iq = icassoShow(sR, 'L', numOfIC, 'colorlimit', [.8 .9]);


[metric_Q, A, W, icasig] = getStableRunEstimates(sR, minClusterSize, maxClusterSize);


function [metric_Q, A, W, icasig, stableRun] = getStableRunEstimates(sR, minClusterSize, maxClusterSize)
%% Get stable run based on code by Sai Ma. Stable run estimates will be used instead of centrotype
%

% number of runs and ICs
numOfRun = length(sR.W);
numOfIC = size(sR.W{1},1);

% Get the centrotype for each cluster and Iq
index2centrotypes = icassoIdx2Centrotype(sR,'partition', sR.cluster.partition(numOfIC,:));
Iq = icassoStability(sR, numOfIC, 'none');

% Find IC index  within each cluster
partition = sR.cluster.partition(numOfIC, :);
clusterindex = cell(1, numOfIC);
for i = 1:numOfIC
    temp = (partition == i);
    clusterindex{i} = sR.index(temp, :);
    clear temp;
end

% Compute stability metric for each run within each cluster
eachRun = zeros(numOfRun, numOfIC);
qc = 0; % num of qualified clusters
for i = 1:numOfIC
    thisCluster = (clusterindex{i}(:,1))';
    clusterSize = length(clusterindex{i});
    if ((clusterSize >= minClusterSize) && (clusterSize <= maxClusterSize) && (Iq(i)>=0.7))
        qc = qc + 1;
        for k = 1:numOfRun
            thisRun = find(thisCluster == k);
            ICindex = (clusterindex{i}(thisRun,1)-1)*numOfIC + clusterindex{i}(thisRun,2);
            if ~isempty(thisRun)
                eachRun(k,i) = max(sR.cluster.similarity(index2centrotypes(i),ICindex'));
            end
            clear thisRun ICindex;
        end
    end
    clear thisCluster clusterSize;
end

%% Find stable run
metric_Q = sum(eachRun,2)/qc;
[dd, stableRun] = max(metric_Q);

%% Get stable run estimates
W = sR.W{stableRun};
clusters_stablerun = partition((stableRun - 1)*numOfIC + 1 : stableRun*numOfIC);
[dd, inds] = sort(clusters_stablerun);
W = W(inds, :);
A = pinv(W);
icasig = W*sR.signal;