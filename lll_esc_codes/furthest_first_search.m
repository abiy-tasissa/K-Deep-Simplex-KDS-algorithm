function [T, C] = furthest_first_search(X, k, lambda, init)
%FURTHEST_FIRST_SEARCH Performs exemplar selection by minimizing
%self-representation cost
%   This code implements the exemplar selection algorithm presented in
% 
%   Chong You, Chi Li, Daniel Robinson, Rene Vidal,
%   "A Scalable Exemplar-based Subspace Clustering Algorithm for 
%   Class-Imbalanced Data", ECCV 2018.
% 
% 	It selects a subset X_0 from the given data X = [x_1, \dots, x_N] that
% 	minimizes the following cost function
%   F_\lambda(X_0) := \max_{j=1,...,N}min_{c_j}||c_j||_1 + lambda/2 ||x_j -
%   X_0 c_j||_2^2 (*)
%   Since (*) is hard to optimize, we compute exemplars by the FFS which  
%   iteratively selecting the worst represented point (see Algorithm 1 in 
%   the paper).
%   This code implements an efficient version of Algorithm 1 as described
%   in the paper (i.e., Algorithm 2)

% Input Arguments
% X                 -- matrix of D by N where each column is a data point.
% k                 -- integer in the range of [1, N] specifying the number
%                      of exemplars to be selected
% lambda            -- model parameter, see Eq. (*)
% init              -- a structure that contains either the field 'T0',
%                      which is the initial set of exemplars 
%                      (e.g. init.T0 = [1, 2]) or the field 'size', which
%                      is the initize of the set of exemplars (e.g., 
%                      init.size = 1 means selecting at random from X one 
%                      point as the initial set of exemplars.)
% Output Arguments
% T                 -- an index set of size k specifying the selected
%                      exemplars
% C                 -- the matrix of size k by N where each column c_j is
%                      the solution to 
%                      min_{c_j}||c_j||_1 + lambda/2 ||x_j - X_0 c_j||_2^2


% Copyright Chong You @ Johns Hopkins University, 2018
% chong.you1987@gmail.com

[D, N] = size(X);

if isfield(init, 'T0')
    init.size = length(init.T0);
end

% initialization
if isfield(init, 'T0')
    T = init.T0;
else
    T = randperm(N, init.size);
end
X_T = X(:, T); % line 1, Algorithm 2

cost = lasso_cost(X_T, X, lambda); % line 2, Algorithm 2

% selection
for ii = init.size + 1: k % line 3, Algorithm 2: select one exemplar in each iteration 
    ii
    [~, ord] = sort(cost, 'descend'); % line 4, Algorithm 2
    max_cost = 0; % line 5, Algorithm 2
    max_cost_idx = [];
    for jj = 1:N % line 6, Algorithm 2
        idx = ord(jj); % line 7, Algorithm 2: update cost(idx) in the following if-else sentence
        if any(T == idx) 
            cost(idx) = 1 - 0.5/lambda; % cost(idx) can be computed in closed form
        else
            cost(idx) = lasso_cost(X_T, X(:, idx), lambda);
        end
        if cost(idx) > max_cost % line 8, Algorithm 2
            max_cost = cost(idx); % line 9, Algorithm 2
            max_cost_idx = idx;
        end % line 10, Algorithm 2
        if max_cost >= cost(ord(jj + 1)) % line 11, Algorithm 2
            break; % line 12, Algorithm 2
        end % line 13, Algorithm 2
    end % line 14, Algorithm 2
    assert(all(max_cost_idx ~= T));
    T = [T, max_cost_idx];
    X_T = [X_T, X(:, max_cost_idx)]; % line 15, Algorithm 2
end % line 16, Algorithm 2

C(1:k, :) = solve_lasso(X_T, X, lambda);
% the representation coefficients for data points in T can be computed in 
% closed form, so we overwrite the corresponding entries of C in the next line:
C(1:k, T) = eye(ii) * (1-1/lambda); 
end

function cost = lasso_cost(X, Y, lambda)
    C = solve_lasso(X, Y, lambda);
    R = Y - X * C;
    cost = sum(abs(C), 1) + lambda / 2 * sum(R .^2, 1);
end

function C = solve_lasso(X, Y,  lambda)
%  This function solves the following lasso problem
%  min_{c_j}||c_j||_1 + lambda/2 ||y_j - X c_j||_2^2
%  where y_j is the j-th column of Y. The output is C = [c_1, c_2, ...]
%  We use the SPAMS package http://spams-devel.gforge.inria.fr/ for solving
%  the lasso problem. You can also replace it with your favorate solver.
param.lambda = 1 / lambda;
C = mexLasso(Y, X, param); 
% sz_X = size(X);
% sz_Y = size(Y);
% C = zeros(sz_X(2),sz_Y(2));
% sz_C = size(C);
% for i = 1:sz_C(2)
%     %[C(:,i),~,~] = l1_ls(X,Y(:,i),lambda);
%     C(:,i) = lasso(X,Y(:,i),'Lambda',lambda);
%     %C(:,i) = lasso_admm(X,Y(:,i),lambda,1,1);
% end
%C = lasso(X,Y(:,1));
end
