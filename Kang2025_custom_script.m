%--------------------------------------------------------------------------
% Custom MATLAB script (Kang et al., 2025, Communications Biology)
%--------------------------------------------------------------------------
clear;clc;

%% --------------------------------------------------------------------------
% Representational Similarity Analysis (RSA)
%--------------------------------------------------------------------------
nTrials = 55;% number of trials
nROIs = 2;% number of target ROIs
ROISizes = [300,400];% sizes (number of voxels) of each ROI

% load behavior data
load('datafile_behav.mat', 'behavPCMatrix'); % matrix size: nTrials x nPCs

for cTrial1 = 1:nTrials
    for cTrial2 = 1:nTrials
        % calculate distances on each PC dimension
        Model_PC1 = abs(behavPCMatrix(cTrial1, 1) - behavPCMatrix(cTrial2, 1));
        Model_PC2 = abs(behavPCMatrix(cTrial1, 2) - behavPCMatrix(cTrial2, 2));
    end
end
offdiag_I = logical(ones(5)-eye(5));
Model_PC1 = Model_PC1(offdiag_I); Model_PC2 = Model_PC2(offdiag_I);

% normalize by dividing the distances by the maximum value of each dimension
Model_PC1 = Model_PC1 / max(max(Model_PC1));
Model_PC2 = Model_PC2 / max(max(Model_PC2));


% load fMRI data
load('datafile_fMRI.mat','patternMatrix'); % matrix size: nTrias x max(ROISizes) x nROIs

for cROI = 1:nROIs
    % derive pattern correlation matrix
    tempPatterns = squeeze(patternMatrix(:, 1:ROIsizes(cROI), cROI));
    tempCorr = corrcoef(tempPatterns);

    % Fisher's Z transformation, z = 0.5 * log((1+r)/(1-r))
    trans_tempCorr = 0.5 * log((1+tempCorr)./(1-tempCorr));
    trans_tempCorr = trans_tempCorr(offdiag_I);

    % calculate correlation with PC distance matrices
    [r1, ~] = corr(-trans_tempCorr, Model_PC1, 'type','Spearman');
    r1 = 0.5*log((1+r1)/(1-r1));
    [r2, ~] = corr(-trans_tempCorr, Model_PC2, 'type','Spearman');
    r2 = 0.5*log((1+r2)/(1-r2));

    RSAResult(cROI, 1) = r1; RSAResult(cROI, 2) = r2;
end

% save data
saveName = 'RSA_data.mat';
save(saveName,'RSAResult');


%% --------------------------------------------------------------------------
% Support Vector Regression
%--------------------------------------------------------------------------
nTrials = 55;% number of trials
nROIs = 2;% number of target ROIs
ROISizes = [300,400,250,550];% sizes (number of voxels) of each ROI
nPCs = 37;% number of features for the SVR model
nRepeats = 100;% number of repeats
nFolds = 10;% number of folds

% load behavior data
load('datafile_behav.mat', 'behavPCMatrix'); % matrix size: nTrials x nPCs

% load fMRI data 
load('datafile_fMRI.mat','patternMatrix'); % matrix size: nTrias x max(ROISizes) x nROIs

for cROI = 1:nROIs
    % derive pattern correlation matrix
    tempPatterns = squeeze(patternMatrix(:, 1:ROIsizes(cROI), cROI));

    % derive feature for the SVR model
    [tempCoeff, ~, ~] = pca(tempPatterns);
    tempPCPatterns = tempPatterns*tempCoeff;
    tempFeatures = tempPCPatterns(:, 1:nPCs);

    for cRepeat = 1:nRepeats
        % execute 10-Fold cross-validation
        cv = cvpartition(nTrials, 'KFold', nFolds);
        for cFold = 1:nFolds
            trainIdx = training(cv, cFold); testIdx = test(cv, cFold);

            % train each SVR model to predict PC values
            svrModel1 = fitrsvm(tempFeatures(trainIdx, :), behavPCMatrix(trainIdx, 1));
            svrModel2 = fitrsvm(tempFeatures(trainIdx, :), behavPCMatrix(trainIdx, 2));

            % test each model on test trials
            predictPC1(testIdx) = predict(svrModel1, tempFeatures(testIdx, :));
            predictPC2(testIdx) = predict(svrModel2, tempFeatures(testIdx, :));
        end
        % calculate correlation between predicted PC values and observed PC values
        [r1, ~] = corr(predictPC1', behavPCMatrix(:, 1), 'type','Spearman');
        r1 = 0.5*log((1+r1)/(1-r1));
        tempResult1(cRepeat) = r1;
        [r2, ~] = corr(predictPC2', behavPCMatrix(:, 2), 'type','Spearman');
        r2 = 0.5*log((1+r2)/(1-r2));
        tempResult2(cRepeat) = r2;
    end
    % derive avergae correlation coefficient
    SVRResult(cROI, 1) = mean(tempResult1);
    SVRResult(cROI, 2) = mean(tempResult2);
end

% save data
saveName = 'SVR_data.mat';
save(saveName,'SVRResult');


%% --------------------------------------------------------------------------
% Representational Connectivity Analysis with RSA
%--------------------------------------------------------------------------

% Representational connectivity analysis between ROI 1 and ROI 2
ROI1 = 1; ROI2 = 2;% ROI indices
nROIs = 2;% number of target ROIs
ROISizes = [300,400,250,500];% sizes (number of voxels) of each ROI

% load fMRI data of task A and B
load('datafilename.mat','patternMatrix'); % matrix size: nConds x max(ROISizes) x nROIs

ROI1_patternMatrix = squeeze(patternMatrix(:,1:ROISizes(1),1));
ROI2_patternMatrix = squeeze(patternMatrix(:,1:ROISizes(2),2));

% derive pattern correlation matrix
tempCorr1 = corrcoef(ROI1_patternMatrix);
tempCorr2 = corrcoef(ROI2_patternMatrix);

% Fisher transformatio z = 0.5 * log((1+r)/(1-r))
trans_Corr1 = 0.5 * log((1+tempCorr1)./(1-tempCorr1));
trans_Corr2 = 0.5 * log((1+tempCorr2)./(1-tempCorr2));

% derive RDM (representational dissimilarity matrix) for each ROI
offdiag_I = logical(ones(5)-eye(5));
RSM_ROI1 = trans_Corr1;
RSM_ROI2 = trans_Corr2;
RDM_ROI1 = - RSM_ROI1(offdiag_I);
RDM_ROI2 = - RSM_ROI2(offdiag_I);

% calculate representational similarity between two ROIs
tmp_corr = corr(RDM_ROI1,RDM_ROI2);

% Fisher transformation z = 0.5 * log((1+r)/(1-r))
trans_corr = 0.5 * log((1+tmp_corr)./(1-tmp_corr));% diagonal --> Inf
crossROI_RS = trans_corr;

% save data
saveName = 'Cross-region-RSA_data.mat';
save(saveName,'crossROI_RS');


%% --------------------------------------------------------------------------
% Representational Connectivity Analysis with SVR
%--------------------------------------------------------------------------
nTrials = 55;% number of trials
seedROInum = 1; targetROInum = 2;% seed ROI, target ROI indices
ROISizes = [300,400,250,550];% sizes (number of voxels) of each ROI

% load fMRI data 
load('datafile_fMRI.mat','patternMatrix'); % matrix size: nTrias x max(ROISizes) x nROIs

% derive pattern correlation matrix
seedPatterns = squeeze(patternMatrix(:, 1:ROISizes(seedROInum), seedROInum));
targetPatterns = squeeze(patternMatrix(:, 1:ROISizes(targetROInum), targetROInum));

% execute Leave-One Trial-Out cross-validation
cv = cvpartition(nTrials, 'LeaveOut');
for cTrial = 1:nTrials
    trainIdx = training(cv, cTrial); testIdx = test(cv, cTrial);

    % train each SVR model to predict voxel t-values
    for cVoxel = 1:ROISizes(targetROInum)
        svrModel = fitrsvm(seedPatterns(trainIdx, :), targetPatterns(trainIdx, 1));
        predictPattern(testIdx, cVoxel) = predict(svrModel, seedPatterns(testIdx, :));
    end
    % calculate correlation between predicted voxel-wise pattern and observed voxel-wise patterns
    [r, ~] = corr(predictPattern(testIdx, :), predictPattern(testIdx, :), 'type','Spearman');
    r = 0.5*log((1+r)/(1-r));
    tempResult(cTrial) = r;
end
% derive avergae correlation coefficient
RCAResult = mean(tempResult);

% save data
saveName = 'RCA_data.mat';
save(saveName,'RCAResult');