function accuracy = testClassificationAccuracy(tuneData, data, chromosome, usePCA, nPCs)
    % Evaluates the fitness of the given candidate solution using the given
    % data.
    
    if nargin < 5
        nPCs = 200;
        nFolds = 5;
        if nargin < 4
            usePCA = false;
        end        
    end

    % Choose random subset of data to compute features for
    nSamples = length(data);
    
    % Evaluate features of data using given chromosome
    if usePCA
        features = zeros(nSamples, nPCs);
        [~, V, ~] = calculatePCA(vertcat(data.image)');
        [hold, ~] = reduceDimension(vertcat(data.image)', V, nPCs);
        features(:, :) = hold';
    else
        % Tune autoencoder using L-BFGS
        tuneAutoEncoder(tuneData, chromosome);
        features = zeros(nSamples, chromosome.hiddenLayerSize);
        features(:, :) = chromosome.computeFeatures(data)';
    end
    labels = [data.label]';
        
    foldAccuracy = zeros(nFolds, 1);
    for i = 1:nFolds
        % Compute train and test ranges of dataset
        testRange = floor((i-1)*nSamples/nFolds)+1:floor(i*nSamples/nFolds);
        trainRange = 1:nSamples;
        trainRange(testRange) = [];
        
        % Train SVM on trainRange
        model = trainSVMinstances(labels(trainRange), features(trainRange, :), '');
        
        % Test SVM on testRange
        foldAccuracy(i) = testSVMinstances(model, labels(testRange),...
                                           features(testRange, :), '');
    end
    
    accuracy = mean(foldAccuracy);
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