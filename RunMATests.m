function RunMATests()
    if ~exist('ma')
        load('ma_run_11_18_2013.mat');
        load('mnist_all.mat');
    end

    if ~exist('data')
        for i = 1:length(trainLabels)
            data(i).label = trainLabels(i);
            data(i).image = trainSamples(i,:);
        end
        for i = 1:length(valLabels)
            valData(i).label = valLabels(i);
            valData(i).image = valSamples(i,:);
        end
    end

    % PCA
    [acc, prec, recall, mcc, fmeas] = testClassificationAccuracy(valData, data, [], true);
    printPerf('PCA', acc, prec, recall, mcc, fmeas);

    % Random chromo
    [acc, prec, recall, mcc, fmeas] = testClassificationAccuracy(valData, data,...
        MemeticAutoencoderChromosome(ma.nodeInitParams));
    printPerf('Random autoencoder', acc, prec, recall, mcc, fmeas);

    % Top chromo
    [acc, prec, recall, mcc, fmeas] = testClassificationAccuracy(valData, data, ma.bestSolution);
    printPerf('Best solution', acc, prec, recall, mcc, fmeas);
end

function printPerf(prefix, acc, prec, recall, mcc, fmeas)
    fprintf('%s accuracy: %f\n', prefix, acc);
    fprintf('%s precision: %f\n', prefix, prec);
    fprintf('%s recall: %f\n', prefix, recall);
    fprintf('%s Matthews Correlation Coefficient: %f\n', prefix, mcc);
    fprintf('%s f-measure: %f\n', prefix, fmeas);
end