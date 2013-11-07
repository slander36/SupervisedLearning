function [accAfter,accBefore,times] = my_stackedAEExercise(trainData,trainLabels,testData,testLabels,numClasses,numLayers,numNodes)
%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

% inputSize = 28 * 28;
%inputSize=size(trainData,1);
%inputSize = 6;
%numClasses = 10;
%numClasses=3;
%hiddenSizeL1 = 200;    % Layer 1 Hidden Size
%hiddenSizeL2 = 200;    % Layer 2 Hidden Size
%hiddenSizeL1 = 80;    
%hiddenSizeL2 = 80;
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term       
maxIter = 200;
%maxIter = 3;
%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

% Load MNIST database files
% trainData = loadMNISTImages('mnist/train-images-idx3-ubyte');
% trainLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
%  
% trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
% 
% DataWithLabels=[trainData;trainLabels'];
% DataWithLabels=DataWithLabels(:,randperm(size(DataWithLabels,2)));
% mnistTrainData=DataWithLabels(1:end-1,:);
% mnistTrainLabels=DataWithLabels(end,:);
% 
% unlabeledData = mnistTrainData;
% 
% trainData   = mnistTrainData;
% trainLabels = mnistTrainLabels; 
% testData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
% testLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
% 
% testLabels(testLabels == 0) = 10; % Remap 0 to 10

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.



t1 = tic;
%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta




%sae1OptTheta = sae1Theta; 

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = maxIter;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

saeOptThetas = cell(numLayers,1);
inputFeatures = trainData;
for curLayer = 1:numLayers
    
    inputSize = size(inputFeatures,1);
    %  Randomly initialize the parameters
    sae1Theta = initializeParameters(numNodes, inputSize);
    
    [sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                       inputSize, numNodes, ...
                                       lambda, sparsityParam, ...
                                       beta, inputFeatures), ...
                                  sae1Theta, options);










    % -------------------------------------------------------------------------



    %%======================================================================
    %% STEP 2: Train the second sparse autoencoder
    %  This trains the second sparse autoencoder on the first autoencoder
    %  featurse.
    %  If you've correctly implemented sparseAutoencoderCost.m, you don't need
    %  to change anything here.

    [inputFeatures] = feedForwardAutoencoder(sae1OptTheta, numNodes, ...
                                            inputSize, inputFeatures);
    saeOptThetas{curLayer} = sae1OptTheta;
    
                                        
end


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

%  Randomly initialize the parameters
% saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);

softmaxModel = struct;

%options.maxIter = 2;
options.maxIter = 100;
softmaxModel = softmaxTrain(numNodes, numClasses, lambda, ...
                            inputFeatures, trainLabels, options);


saeSoftmaxOptTheta = softmaxModel.optTheta(:);

[pred] = softmaxPredict(softmaxModel, inputFeatures);

acc = mean(trainLabels(:) == pred(:));
fprintf('Softmax layer training Accuracy: %0.3f%%\n', acc * 100);

% [pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
%                           numClasses, netconfig, trainData);
% 
% accTrain = mean(trainLabels(:) == pred(:));
% fprintf('Before Finetuning Training Accuracy: %0.3f%%\n', accTrain * 100);

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

% Initialize the stack using the parameters learned
stack = cell(numLayers,1);
for curLayer = 1:numLayers
    sae1OptTheta = saeOptThetas{curLayer};
    if (curLayer == 1)
       inputSize =  size(trainData,1);
    else
       inputSize = numNodes;
    end
    stack{curLayer}.w = reshape(sae1OptTheta(1:numNodes*inputSize), ...
                     numNodes, inputSize);
    stack{curLayer}.b = sae1OptTheta(2*numNodes*inputSize+1:2*numNodes*inputSize+numNodes);

end

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%


[ stackedAEOptTheta, grad ] = minFunc(@(p) stackedAECost(p,size(trainData,1), numNodes, ...
                                              numClasses, netconfig, ...
                                              lambda,  trainData,trainLabels),...
                                              stackedAETheta,options);













% -------------------------------------------------------------------------

times = toc(t1);

%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set

[pred] = stackedAEPredict(stackedAETheta, size(trainData,1), numNodes, ...
                          numClasses, netconfig, trainData);

accTrain = mean(trainLabels(:) == pred(:));
fprintf('Before Finetuning Training Accuracy: %0.3f%%\n', accTrain * 100);

[pred] = stackedAEPredict(stackedAETheta, size(trainData,1), numNodes, ...
                          numClasses, netconfig, testData);

accBefore = mean(testLabels(:) == pred(:));
%CBefore=confusionmat(testLabels,pred);
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', accBefore * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, size(trainData,1), numNodes, ...
                          numClasses, netconfig, testData);

accAfter = mean(testLabels(:) == pred(:));
%CAfter=confusionmat(testLabels,pred);
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', accAfter * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
end