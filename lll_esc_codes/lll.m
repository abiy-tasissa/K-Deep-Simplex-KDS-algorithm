%[X,Y0,X0,lx,Z] = lll(Y,W,d,L[,kZ,nl])
% LLL algorithm for fast Laplacian Eigenmaps
%
% This computes a fast, approximate Laplacian Eigenmaps embedding for a
% high-dimensional dataset using the Locally Linear Landmarks (LLL) algorithm.
% LLL also provides an out-of-sample mapping, see lllmap.m.
%
% For details about how to select the landmarks (indices lx) and construct the
% reconstruction weights (matrix Z), see functions lmarks.m and lweights.m,
% respectively, inside lll.m. At present we use simple but effective choices
% in these functions: random landmarks and LLE-style weights, respectively.
%
% It should be easy to modify lll.m to solve spectral problems other than
% Laplacian Eigenmaps, say LLE, by constructing the relevant graph Laplacian
% and the matrices A and B below.
%
% In:
%   Y: NxD matrix of N row D-dimensional vectors.
%   W: NxN affinity matrix (usually sparse).
%   d: dimension of the latent space.
%   L: number of landmarks.
%   kZ: number of landmarks each data point uses. Default: d+1.
%   nl: normalised (1) or unnormalised (0) Laplacian. Default: 0.
% Out:
%   X: N x d matrix of N row d-dimensional vectors, the projections of Y.
%   Y0: LxD matrix of L row D-dimensional vectors, the landmarks.
%   X0: L x d matrix of L row d-dimensional vectors, the landmark projections.
%   lx: 1xL list of the indices of the landmarks in Y: Y0=Y(lx,:), X0=X(lx,:).
%   Z: LxN matrix of reconstruction weights.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2013 by Max Vladymyrov and Miguel A. Carreira-Perpinan

function [X,Y0,X0,lx,Z] = lll(Y,W,d,L,kZ,nl)

% ---------- Argument defaults ----------
if ~exist('kZ','var') || isempty(kZ) kZ = d+1; end
if ~exist('nl','var') || isempty(nl) nl = 0; end
% ---------- End of "argument defaults" ----------

% Find landmarks and reconstruction weights. See below functions lmarks.m and
% lweights.m (which admit several optional arguments).
%[Y0,lx] = lmarks(Y,L,'kmeans');		% Find L landmarks at random
[Y0,lx] = lmarks(Y,L);		% Find L landmarks at random
Z = lweights(Y,Y0,kZ);		% Find reconstruction weights

% Affinity matrix A for the reduced spectral problem (landmarks-only)
D = sum(W,2); L = diag(D)-W; A = Z*L*Z'; A = (A+A')/2;

% Constraint matrix B for the reduced spectral problem (landmarks-only)
if nl==0 B = Z*Z'; else B = Z*diag(D)*Z'; end; B = (B+B')/2;

% Solve the reduced spectral problem. We can do this either by solving a
% generalized or a standard eigenvalue problem. They have a similar runtime.
% The solution through the generalized eigenvalue problem:
opts.issym = 1; opts.isreal = 1; opts.disp = 0;
%[U,E] = eigs(A,B,d+1,'sm',opts); E = diag(E); [~,ind] = sort(E);
[U,E] = eigs(A,B,d+1,eps,opts); E = diag(E); [~,ind] = sort(E);
X0 = U(:,ind(2:d+1));


% $$$ % The solution through the standard eigenvalue problem:
% $$$ BB = sqrtm(full(B)); C = (BB\A)/BB; C = (C+C')/2;
% $$$ [U,E] = eig(C); E = diag(E); [~,ind] = sort(E); U = U(:,ind(2:d+1));
% $$$ X0 = BB\U;

% Construct the projections for the rest of the points (out-of-sample mapping)
X = Z'*X0;
end

