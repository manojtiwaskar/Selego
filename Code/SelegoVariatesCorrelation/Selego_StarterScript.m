%  Note: To make this Selego code run, you will have to write in total 2 directory path in the following files:
%  In Selego_StarterScript.m : 1. filename variable = Input data file path 
%                       2. outpath variable = Output path for correlation matrix
% we assume that the target variate is the last column of dataset and drop the date/time column from the dataset
% because feature extraction is done variates and not date/time.

filename = '-----input/data/path/----------';
outpath = '-----output/directory/path/--------';

% Hyperparameters for Selego
octaves = 3;
scale = 3;
sigma_t = 0.5;


% Default rank_technique is 'KNN', the other options are 'PPR'
% (Personalized Page Rank) and 'PR' (Page Rank)
rank_technique = 'KNN'; 

% If the rank_technique= 'KNN' or rank_technique= 'PPR' , then provide appropriate seed/target node column number from input data 
seed = -1;


A = readtable(filename);

% generateDataAviage() returns the cell of extracted features for each
% input variates in the dataset
result = generateDataAviage(A, octaves, scale, sigma_t);

% Corr_Mat variable is the Selego Correlation matrix of all the variates in the dataset.
if strcmp(rank_technique, 'KNN')
    ranking = generateGraph_KNN(result, seed);
elseif strcmp(rank_technique, 'PPR') || strcmp(rank_technique, 'PR')
    ranking = generateGraph_PR_PPR(result,rank_technique, seed);
end


% Write the correlation matrix at path mentioned in outpath variable.
writematrix(ranking, outpath);

