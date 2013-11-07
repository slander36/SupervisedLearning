function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data,2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
[n,m]=size(data);
M=theta*data;
M=bsxfun(@minus,M,max(M,[],1));
N=exp(M);
N=bsxfun(@rdivide,N,sum(N));
h=N;

%cost=-sum(log(h(sub2ind(size(h), labels', 1:length(labels')))))/m;

cost=-sum(sum(groundTruth.*log(h)))/m;
cost=cost+(lambda/2)*sum(sum(theta.*theta));
thetagrad=-((groundTruth-h)*data')/m;
thetagrad=thetagrad+lambda*theta;

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

