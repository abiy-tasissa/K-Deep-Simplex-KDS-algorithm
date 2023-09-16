% -----------------------------------------------------------------------------------
% This script loads the optimal sparse coefficients from SMCE and runs
% spectral clustering on them. Since SMCE on MNIST-5 could be potentially
% expensive, this allows the user to directly obtain the clustering without
% doing SMCE optimization. The user can also use the sparse representation
% coefficients for further analysis.
% The parameter choices are: (1) lambda = 1, (2) lambda = 10, (3) lambda = 100 and 
% (4) lambda = 200. Lambda is a regularization term that controls locality 
% regularization in SMCE. We also consider three types of pre-processing: 
% (1) Scaling to[0,1], (2) Standardizing and (3) Normalizing. 
% Example: `mnist_accuracy_3_4.mat` loads the sparse coefficient matrix 
% with pre-processing set to (3) and lambda set to 400.
% -----------------------------------------------------------------------------------
% Add path to SMCE folder
addpath(genpath('./SMCE'))
addpath(genpath('./mnist_KDS_data'))
% Load MNIST dataset and save ground truth
load("./datasets/mnist_03467.mat");
% Specify number of data points
num_data = length(labels);
% Data and labels
gtruth = double(labels);
gtruth = gtruth';
% Load the desired spase coefficient. 
load('mnist_accuracy_1_2.mat');
% symmetrize the adjacency matrices
W = processC_fast(W,0.95);
Wsym = max(abs(W),abs(W)');
% perform clustering
n = 5;
[C, ~, ~] = SpectralClustering_fast(Wsym, n, 3);
missrate = missclassGroups(C,gtruth+1)/length(gtruth);
