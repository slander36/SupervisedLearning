function [accuracy, precision, recall, mcc, fmeasure] = testClassificationAccuracy(...
    tuneData, data, chromosome, usePCA, nPCs)
    % Evaluates the fitness of the given candidate solution using the given
    % data.
    
    nFolds = 5;
    if nargin < 5
        nPCs = 200;
        if nargin < 4
            usePCA = false;
        end        
    end

    % Choose random subset of data to compute features for
    nSamples = length(data);
    
    % Evaluate features of data using given chromosome
    if usePCA
        features = zeros(nSamples, nPCs);
        [~, V, ~] = calculatePCA(vertcat(tuneData.image)');
        [hold, ~] = reduceDimension(vertcat(tuneData.image)', V, nPCs);
        features(:, :) = hold';
    else
        % Tune autoencoder using L-BFGS
        tuneAutoEncoder(tuneData, chromosome);
        features = zeros(nSamples, chromosome.hiddenLayerSize);
        features(:, :) = chromosome.computeFeatures(data)';
    end
    labels = [data.label]';
        
    tp = 0;
    fp = 0;
    fn = 0;
    tn = 0;
    for i = 1:nFolds
        % Compute train and test ranges of dataset
        testRange = floor((i-1)*nSamples/nFolds)+1:floor(i*nSamples/nFolds);
        trainRange = 1:nSamples;
        trainRange(testRange) = [];
        
        % Train SVM on trainRange
        model = trainSVMinstances(labels(trainRange), features(trainRange, :), '');
        
        % Test SVM on testRange
        [~, predictedLabels] = testSVMinstances(model,...
            labels(testRange), features(testRange, :), '');
        tp = tp + length(intersect(find(predictedLabels==1),...
                                   find(predictedLabels==labels(testRange))));
        fp = fp + length(intersect(find(predictedLabels==1),...
                                   find(predictedLabels~=labels(testRange))));
        tn = tn + length(intersect(find(predictedLabels==0),...
                                   find(predictedLabels==labels(testRange))));
        fn = fn + length(intersect(find(predictedLabels==0),...
                                   find(predictedLabels~=labels(testRange))));
    end
    
    accuracy = (tp+tn)/(tp+tn+fp+fn);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    % Matthews correlation coefficient
    mcc = (tp*tn - fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));
    fmeasure = 2*(precision*recall)/(precision+recall);
end


function [m, V, D] = calculatePCA(samples)
    %Calculate cov matrix and the PCA matrixes
    [r,c] = size(samples);
    m           = mean(samples')';
    S			= ((samples - m*ones(1,c)) * (samples - m*ones(1,c))');
    [V, D]	    = eig(S);
end

function [reducedSamples, W] = reduceDimension(samples, V, dimension)
    %Create PCA transformation matrix from full eigenmatrix
    W = V(:,size(samples, 1)-dimension+1:size(samples, 1))';
    
    reducedSamples = W*samples;
end