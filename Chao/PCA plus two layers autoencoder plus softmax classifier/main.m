rng(1,'twister');
tStart=cputime;
mnistData = loadMNISTImages('mnist/train-images-idx3-ubyte');
mnistLabels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
mnistTestData = loadMNISTImages('mnist/t10k-images-idx3-ubyte');
mnistTestLabels = loadMNISTLabels('mnist/t10k-labels-idx1-ubyte');
mnistLabels=mnistLabels'+1;
mnistTestLabels=mnistTestLabels'+1;

%retainRate=0.80;
%retainRate=0.90;
retainRate=0.95;
%retainRate=0.99;
%retainRate=1;

[ xTrainZCAwhite,xTrainPCAwhite,xTrainPCA,k,U,S ] = getPCAWhiteAndZCAWhite( mnistData,retainRate );


[ xTestZCAwhite,xTestPCAwhite,xTestPCA ] = processTestData( mnistTestData,U,k,S );


accTrain=zeros(10,1);
accBeforePCA=zeros(10,1);
accAfterPCA=zeros(10,1);

% CBeforeFinetuningPCAwhite=cell(10,1);
% CAfterFinetuningPCAwhite=cell(10,1);
% accBeforeFinetuningPCAwhite=cell(10,1);
% accAfterFinetuningPCAwhite=cell(10,1);


% CBeforeZCAwhite=cell(10,1);
% CAfterZCAwhite=cell(10,1);
% accBeforeZCAwhite=cell(10,1);
% accAfterZCAwhite=cell(10,1);

xTrainPCA=xTrainPCA(:,1:1000);
mnistLabels=mnistLabels(:,1:1000);
xTestPCA=xTestPCA(:,1:100);
mnistTestLabels=mnistTestLabels(:,1:100);

for i=1:1
    [accTrain(i,1),accBeforePCA(i,1),accAfterPCA(i,1)] = stackedAEExercise(xTrainPCA,mnistLabels,xTestPCA,mnistTestLabels);    
    %[CBeforeFinetuningPCAwhite{i,1},CAfterFinetuningPCAwhite{i,1},accBeforeFinetuningPCAwhite{i,1},accAfterFinetuningPCAwhite{i,1}] = stackedAEExercise(xTrainPCAwhite,mnistLabels,xTestPCAwhite,mnistTestLabels);
    %[CBeforeZCAwhite{i,1},CAfterZCAwhite{i,1},accBeforeZCAwhite{i,1},accAfterZCAwhite{i,1}] = stackedAEExercise(xTrainZCAwhite,mnistLabels,xTestZCAwhite,mnistTestLabels);
    
end
tEnd=cputime;