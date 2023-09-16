% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------
function C = fast_embedding(Z,k)
sz_Z = size(Z);
n = sz_Z(2);
W = Z*Z';
% Degree matrix
degs=  sum(W,2);
D    = sparse(1:size(W, 1), 1:size(W, 2), degs);
% compute unnormalized Laplacian
L = diag(sum(W,2)) - W;
degs(degs == 0) = eps;
% calculate D^(-1/2)
D = diag(sum(W,2));
D = spdiags(1./(degs.^0.5), 0, size(D, 1), size(D, 2));
        
% calculate normalized Laplacian
L = D * L * D;
L = 0.5*(L+L');
% 
% compute the eigenvectors corresponding to the k smallest
% eigenvalues
diff   = eps;
[U1, ~] = eigs(L, k, diff);
U1 = bsxfun(@rdivide, U1, sqrt(sum(U1.^2, 2)));
U2 = Z'*U1;
U = [U2;U1];
C = kmeans(U, k, 'EmptyAction', 'singleton','Replicates',50,'MaxIter',1000);
C = C(1:m);
end