% Driver script for MemeticAutoencoder algorithm

buildData = true;

if buildData
    for i = 1:length(trainLabels)
        data(i).label = trainLabels(i);
        data(i).image = trainSamples(i,:);
    end
end

ma = MemeticAutoencoder();
[bestSolution, topFitnessValues] = ma.trainMemeticAutoencoder(data(1:10000));
