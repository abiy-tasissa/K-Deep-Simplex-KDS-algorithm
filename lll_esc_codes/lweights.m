% Z = lweights(Y,Y0,kZ,[sqd,method]) Reconstruction weights from landmarks
%
% This computes the reconstruction weights of each data point from its kZ
% nearest landmarks.
%
% Finding the nearest landmarks is done by default by sorting L distances for
% each point. If k<<L, a faster way is to use selection in O(L+k.logk) rather
% than sorting in O(L.logL). To use selection, install this package and set
% method='mink':
%   Min/Max selection
%   http://www.mathworks.com/matlabcentral/fileexchange/23576-minmax-selection
% Its mink() function selects the k smallest elements in the array with a
% partial quicksort in O(L+k.logk).
% Note that computing the distances between pairs of points is O(D.L²) anyway,
% so the improvement of selection is larger the smaller D is.
%
% In:
%   Y: NxD matrix of N row D-dimensional vectors, the dataset.
%   Y0: LxD matrix of L row D-dimensional vectors, the landmarks.
%   kZ: number of neighbouring landmarks to use for each data point.
%   sqd: NxL matrix of pairwise distances between points and landmarks.
%   method: how to compute the neighboring landmarks ('min_min','sort','mink').
%      Default: 'min_min' if kZ <= 5, otherwise 'sort'.
% Out:
%   Z: LxN matrix where Z(:,n) are the reconstruction weights for data point n.
%
% Any non-mandatory argument can be given the value [] to force it to take
% its default value.

% Copyright (c) 2013 by Max Vladymyrov and Miguel A. Carreira-Perpinan

function Z = lweights(Y,Y0,kZ,sqd,method)

% ---------- Argument defaults ----------
if ~exist('sqd','var') || isempty(sqd) sqd = sqdist(Y0,Y); end
if ~exist('method','var') || isempty(method)
  if kZ>5 method = 'sort'; else method = 'min_min'; end
end
% ---------- End of "argument defaults" ----------

N = size(Y,1); [L,D] = size(Y0);

switch method
 case 'min_min'
  k_ind = zeros(kZ,N);
  for i=1:kZ
    [~,k_ind(i,:)] = min(sqd); sqd(sub2ind(size(sqd),k_ind(i,:),1:N)) = Inf;
  end
 case 'sort'
  [~,ind] = sort(sqd); k_ind = ind(1:kZ,:);
 case 'mex'
  [~,ind] = mink(sqd,kZ); k_ind = ind;
end
clear sqd

T = zeros(kZ,N); kZ1 = ones(kZ,1);
for i=1:N
  dY = bsxfun(@minus,Y(i,:),Y0(k_ind(:,i),:)); C = dY*dY'; % local covariance
  if L>D 
      C = C + eye(kZ,kZ)*1e-10*sum(diag(C)); 
  end
  % regularize covariance
  zi = C\kZ1;                                   % Solve linear system
  zi = zi/sum(zi);                              % Make sure that sum is 1
  T(:,i) = zi;
end

Z = sparse(k_ind(:),floor(((1:N*kZ)+kZ-1)/kZ),T(:),L,N,kZ*N);

