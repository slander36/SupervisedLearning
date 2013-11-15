function fitness = evaluateMAFitness(data, chromo, nFolds)
    % Evaluates the fitness of the given chromosome (weight matrix) using the
    % given data

    if nargin < 3
        nFolds = 5;
    end

    % Do L-BFGS on the chromosome for a few iterations

    % Set parameters to L-BFGS
    sparsityParam = 0.1;   % desired average activation of the hidden units.
                           % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                                   %  in the lecture notes). 
    lambda = 3e-3;         % weight decay parameter       
    beta = 3;              % weight of sparsity penalty term       
    maxIter = 200
    inputSize = size(data, 2); % TODO: correct size of input layer

    % TODO: build trainData
    trainData = [];
    autoEncoder = chromo.autoEncoder;

    %  Use minFunc to minimize the function
    addpath minFunc/
    options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                              % function. Generally, for minFunc to work, you
                              % need a function pointer with two outputs: the
                              % function value and the gradient. In our problem,
                              % sparseAutoencoderCost.m satisfies this.
    options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
    options.display = 'on';


    [autoEncoderTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                       inputSize, length(autoencoder), ...
                                       lambda, sparsityParam, ...
                                       beta, trainData), ...
                                       autoEncoder, options);

    % Evaluate performance of tuned auto encoder matrix using cross-fold
    % validation classification performance using an SVM
    accuracy = zeros(nFolds, 1);
    for i = 1:nFolds
        % Build train and test datasets
        % TODO

        % Compute features for this fold's training set
        features = feedForwardAutoencoder(autoEncoderTheta, length(autoEncoderTheta),
                                          size(data, 2), trainData);
    end

    % Compute fitness as the average classification accuracy
    fitness = mean(accuracy);
end
