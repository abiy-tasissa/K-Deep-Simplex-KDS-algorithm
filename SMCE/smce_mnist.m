%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2012
%--------------------------------------------------------------------------

function missrate = smce_mnist(Y,lambda,KMax,dim,n,gtruth,verbose)

if (nargin < 7)
    verbose = true;
end
if (nargin < 6)
    gtruth = [];
end
if (nargin < 5)
    n = 1;
end
if (nargin < 4)
    dim  = 2;
end

% solve the sparse optimization program
W = smce_optimization_fast(Y,lambda,KMax,verbose);
clear data
clear Y
% symmetrize the adjacency matrices
W = processC_fast(W,0.95);
Wsym = max(abs(W),abs(W)');
% perform clustering
n = 5;
[C, ~, ~] = SpectralClustering_fast(Wsym, n, 3);
missrate = missclassGroups(C,gtruth+1)/length(gtruth);

end