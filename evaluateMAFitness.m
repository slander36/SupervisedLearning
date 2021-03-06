function [fitness, error] = evaluateMAFitness(data, chromo, useLBFGS)
    % Evaluates the fitness of the given chromosome (weight matrix) using the
    % given data

    if nargin < 3
        useLBFGS = true;
    end

    % Do L-BFGS on the chromosome for a few iterations

    % Set parameters to L-BFGS
    sparsityParam = 0.1;   % desired average activation of the hidden units.
                           % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                                   %  in the lecture notes). 
    lambda = 3e-3;         % weight decay parameter       
    beta = 3;              % weight of sparsity penalty term       
    maxIter = 2;

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

    if useLBFGS
        % Run L-BFGS on the autoEncoder 
        [autoEncoderTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                           chromo.inputLayerSize, chromo.hiddenLayerSize, ...
                                           lambda, sparsityParam, ...
                                           beta, vertcat(data.image)'), ...
                                           autoEncoder, options);
    else
        autoEncoderTheta = autoEncoder;
    end
    
    % Get reconstruction error for all samples
    reconstruction = reconstructFromAutoencoder(autoEncoderTheta, chromo.hiddenLayerSize,...
                                                chromo.inputLayerSize, vertcat(data.image)');
    error = mean(sum((reconstruction - vertcat(data.image)').^2));
    fitness = exp(-error/1000);
end
