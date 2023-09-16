% ------------------------------------------------------------------------------
% This function loads a noisy Salinas A dataset and its corresponding
% labels
% Why noisy Salinas A? The definition of the dictionary for SMCE is based
% on the data matrix and is ill-defined when two columsn of data are nearly
% equal. To disambiguate this, we add a random noise to Salinas A data.
% Note that the same data, for a fixed realization, is used to compare with
% KDS
% -------------------------------------------------------------------------------
function [x,labels] = load_salinas
    % Load SalinasA data
    %load('./datasets/SalinasA.mat');
    load('./datasets/SalinasA_smallNoise.mat');
    load('./datasets/SalinasA_gt.mat');
    % Create the data matrix
    %X = salinasA;
    X = permute(X,[2 1 3]);
    x = reshape(X,7138,224);
    %     x = x+1e-3*randn(size(x));
    % Remove zero labels and re-number labels so that it is [1,6] consistent
    % the way SMCE accepts inputs
    labels = reshape(salinasA_gt,7138,1);
    zero_labels = find(labels==0);
    non_zero_labels = setdiff(1:length(labels),zero_labels);
    x = x(non_zero_labels,:);
    labels = labels(non_zero_labels);
    labels(labels==0)=7;
    labels(labels==10)=2;
    labels(labels==11)=3;
    labels(labels==12)=4;
    labels(labels==13)=5;
    labels(labels==14)=6;
    x = x';
end
