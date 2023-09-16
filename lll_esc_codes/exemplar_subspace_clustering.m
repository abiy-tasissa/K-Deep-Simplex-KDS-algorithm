function groups = exemplar_subspace_clustering(X, nCluster, k, lambda, t)
%EXEMPLAR_SUBSPACE_CLUSTERING Performs exemplar based subspace clustering
%   This code implements the exemplar based subspace clustering method 
%   (Algorithm 3) in 
% 
%   Chong You, Chi Li, Daniel Robinson, Rene Vidal,
%   "A Scalable Exemplar-based Subspace Clustering Algorithm for 
%   Class-Imbalanced Data", ECCV 2018.

% Input Arguments
% X                 -- matrix of D by N where each column is a data point.
% nCluster          -- number of clusters.
% k                 -- integer in the range of [1, N] specifying the number
%                      of exemplars to be selected
% t                 -- integer, number of nearest neighbors
% lambda            -- model parameter, see furthest_first_search.m
%
% Copyright Chong You @ Johns Hopkins University, 2018
% chong.you1987@gmail.com

% Select exemplars (line 1, algorithm 3)
init.size = 1;
%init.size = t;
[~, C0] = furthest_first_search(X, k, lambda, init);
C0 = cnormalize_inplace(full(C0));

% Build affinity (line 2, algorithm 3)
% W = knn_graph(C0, t, 'vlfeat');
W = knn_graph(C0, t, 'default'); % use this line if vlfeat is not installed
A = W + W';

% Spectral clustering (line 3, algorithm 3)
%groups = SpectralClustering(A, nCluster, 'Eig_Solver', 'eigs','Kmeans_Solver', 'vl_kmeans');  
 groups = SpectralClustering(A, nCluster, 'Eig_Solver', 'eigs');   % use this line if vlfeat is not installed