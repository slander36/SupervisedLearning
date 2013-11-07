clc
clear
rng(0);

% load the dataset, a subset of the MNIST digits
load mnist_all.mat

trainSamples = [trainSamples; valSamples]';
testSamples = testSamples';
trainLabels = [trainLabels;valLabels]';
testLabels = testLabels';

% def # of layers and # of units in each layer. Suppose each layer has the
% same # of units
layers = {1 2 3};
nLayers = length(layers);
nodes = {50 100 200 300};
%nodes = {200};
nNodes = length(nodes);
accResults = zeros(nLayers,nNodes);
accBefore = zeros(nLayers,nNodes);
times = zeros(nLayers,nNodes);

for i = 1:nLayers
    kLayer = layers{i};
    for j = 1:nNodes
        kNode = nodes{j};
        fprintf('====== layer: %d node: %d ==========\n',kLayer,kNode);
        [accResults(i,j),accBefore(i,j),times(i,j)] = my_stackedAEExercise(trainSamples,trainLabels,testSamples,testLabels,10,kLayer,kNode);

    end
end
save('accNTimes','accResults','accBefore','times');

