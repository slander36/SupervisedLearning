function features = computeMAFeatures(data, chromo, tuneData)
    % Evaluates the fitness of the given chromosome (weight matrix) using the
    % given data

    if nargin < 3
        useLBFGS = false;
    else
        useLBFGS = true;
    end

    autoEncoder = chromo.autoEncoder;
    if useLBFGS
        % Do L-BFGS on the chromosome for a few iterations

        % Set parameters to L-BFGS
        sparsityParam = 0.1;   % desired average activation of the hidden units.
                               % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
                                       %  in the lecture notes). 
        lambda = 3e-3;         % weight decay parameter       
        beta = 3;              % weight of sparsity penalty term       
        maxIter = 2;

        %  Use minFunc to minimize the function
        addpath minFunc/
        options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                                  % function. Generally, for minFunc to work, you
                                  % need a function pointer with two outputs: the
                                  % function value and the gradient. In our problem,
                                  % sparseAutoencoderCost.m satisfies this.
        options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
        options.display = 'on';
        
        % Run L-BFGS on the autoEncoder 
        [autoEncoderTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                           chromo.inputLayerSize, chromo.hiddenLayerSize, ...
                                           lambda, sparsityParam, ...
                                           beta, vertcat(tuneData.image)'), ...
                                           autoEncoder, options);
    else
        autoEncoderTheta = autoEncoder;
    end
    
    % Get activation of this autoencoder
    features = feedForwardAutoencoder(autoEncoderTheta, chromo.hiddenLayerSize,...
                                                chromo.inputLayerSize, vertcat(data.image)');
end
