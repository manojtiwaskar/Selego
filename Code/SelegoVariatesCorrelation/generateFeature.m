function [frames,gss,dogss] = generateFeature(timeSeries, octaves, scale, sigma_t)

%data= csvread('D:\Motif_Results\Datasets\SynteticDataset\data\IndexEmbeddedFeatures\ORGRWMotif_10_1_85_instance_1_0.csv');%csvread('file.csv');
% data =data(1,:);
data = timeSeries';
DeOctTime = octaves;
DeLevelTime = scale;
% DeSigmaDepd = 0.4;%
DeSigmaTime = sigma_t;%
DeGaussianThres = 0.1;%
% thresh = 0.04 / DeLevelTime / 2 ;%0.04;%
DeSpatialBins = 4; %NUMBER OF BINs
r= 10; %5 threshould variates
sBoundary=1;
eBoundary=size(data',1);
[frames,gss,dogss] = sift_gaussianSmooth_SV(data', DeOctTime, DeLevelTime, DeSigmaTime,...
    DeSpatialBins, DeGaussianThres, r, sBoundary, eBoundary);
