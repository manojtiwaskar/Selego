%  Note: To make this Selego code run, you will have to write in total 2 directory path in the following files:
%  In Selego_StarterScript.m : 1. filename variable = Input data file path 
%                       2. outpath variable = Output path for correlation matrix
% we assume that the target variate is the last column of dataset and drop the date/time column from the dataset
% because feature extraction is done variates and not date/time.

filename = '-----input/data/path/----------';
outpath = '-----output/directory/path/--------';

A = readtable(filename);

% generateDataAviage() returns the cell of extracted features for each
% input variates in the dataset
result = generateDataAviage(A);

% Corr_Mat variable is the Selego Correlation matrix of all the variates in the dataset.
Corr_Mat = generateGraph(result, 1);
T = array2table(Corr_Mat);

% Write the correlation matrix at path mentioned in outpath variable.
writetable(T, outpath);

