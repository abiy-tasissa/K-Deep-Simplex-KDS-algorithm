function R = knn_graph(X, K, NSMethod)
%KNN_GRAPH compute k nearest neighbor graph
%   Given data X = [x_1, ..., x_N], compute the k-NN graph with weight
%   matrix R where r_{ij} = 1 if x_j is a k-NN of x_i, and 0 otherwise.

% Input Arguments
% X                  -- matrix of D by N where each column is a data point
% K                  -- integer, number of nearest neighbors
% NSMethod           -- NSMethod = 'default': use the matlab function
%                       knnsearch; NSMethod = 'vlfeat': use kdtree provided
%                       in the vlfeat toolbox.

% Copyright Chong You @ Johns Hopkins University, 2016
% chong.you1987@gmail.com

N = size(X, 2);
if ~exist('NSMethod', 'var')
    NSMethod = 'default';
end

if strcmpi(NSMethod, 'default')
    IDX = knnsearch(X', X', 'K', K+1);
else
    if strcmpi(NSMethod, 'vlfeat')
        kdtree = vl_kdtreebuild(X);
        IDX = vl_kdtreequery(kdtree, X, X, 'NumNeighbors', K+1)' ;
        IDX = double(IDX);
    else
        IDX = knnsearch(X', X', 'K', K+1, 'NSMethod', NSMethod);
    end
end
R = sparse( repmat(1:N, 1, K), IDX(N+1:end), 1, N, N );
end
