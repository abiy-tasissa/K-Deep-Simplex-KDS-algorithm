% ----------------------------------------------------------------------------------
% This is a demo script that runs ESC experiments from the paper
% ESC is an abbreviation for "exemplar based subspace clustering"
% The algorithm was proposed in the paper referenced below:
% Ref: You, C., Li, C., Robinson, D. P., & Vidal, R. (2018). 
% Scalable exemplar-based subspace clustering on class-imbalanced data. 
% In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 67-83).
% ---------------------------------------------------------------------------------
% Choice of datasets
% choice = 1 is the Extended Yale Face Dataset
% The face dataset can be downloaded from the link below 
% http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
% You can use the load_yale script to prepare the data as input for SMCE
% ---------------------------------------------------------------------------------
% choice = 2 is the Salinas A dataset. This is a hyperspectral dataset
% The source of these datasets is below:
% http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
% You can use the load_salinas script to prepare the data as input for SMCE
% ---------------------------------------------------------------------------------
% choice = 3 is the MNIST digit dataset. For details of these dataset,
% you can read the details here: http://yann.lecun.com/exdb/mnist/
% This data is included along with this script
% Here, we only consider the 5 digits 0,3,4,6,7. 
% --------------------------------------------------------------------------------
% choice = 4 is a synthetic dataset of two moons. This dataset can be
% generated using sklearn in PYTHON
% Reference:
% https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
% This data is included along with this script
% ----------------------------------------------------------------------------------
% Add path to different necessary folders
addpath('./toolbox')
addpath('./spams-matlab-v2.6')
addpath('./datasets')
addpath('./lll_esc_codes/');
start_spams
% Choose data and pre-processing
data_choice = 1;
% We consider three pre-processing techniques
% (1) Scaling to [0,1]
% (2) Standardizing i.e. N(0,1)
% (3) Normalizing each column of the data matrix by its l2 norm
scaling_choice = 1;
if data_choice==1
    [x,labels]=load_yale;
    if scaling_choice==1
        x = x/max(max(x));
    end
    if scaling_choice==2
        x = (x-mean(x))./std(x);
    end
    if scaling_choice==3
        x = x./vecnorm(x);
    end
    % ESC parameters
    % m = number of atoms
    % k = number of clusters
    % t = number of nearest neighbours
    lambda = 200; m = 32; k = 2;
    n = length(unique(labels));
    verbose = true;
    gtruth = labels;
    % Call ESC
    accr_array = zeros(10,1);
    for i = 1:10
        groups = exemplar_subspace_clustering(x, k, m, lambda, 2);
        % Accuracy
        groups = bestMap(gtruth, groups);
        % average over 5 runs
        accr  = sum(gtruth(:) == groups(:)) / length(gtruth);
        accr_array(i) = accr;
    end
end
% ----------------------------------------------------------------------------------
% Salinas A dataset
if data_choice == 2
    [x,labels]=load_salinas;
    if scaling_choice==1
        x = x/max(max(x));
    end
    if scaling_choice==2
        x = (x-mean(x))./std(x);
    end
    if scaling_choice==3
        x = x./vecnorm(x);
    end
    % ESC parameters
    % m = number of atoms
    % k = number of clusters
    % t = number of nearest neighbours
    lambda = 200; m = 600; k = 6;
    gtruth = labels;
    %Average over 5 runs
    accr_array = zeros(5,1);
    for i = 1:5
    % Call ESC
    groups = exemplar_subspace_clustering(x, k, m, lambda,3);
    % Accuracy
    groups = bestMap(gtruth, groups);
    accr  = sum(gtruth(:) == groups(:)) / length(gtruth);
    accr_array(i) = accr;
    end
end
% ------------------------------------------------------------------------------
% MNIST dataset
if data_choice ==3
  % Load 5 digits of MNIST
  load("./datasets/mnist_03467.mat");
  x = double(data);
  % Specify number of data points
  num_data = 35037;
  % Data and labels
  x = x(1:num_data,:);
  x= x';
  gtruth = double(labels);
  gtruth = gtruth+1;
  gtruth = gtruth(1:num_data);
  gtruth = gtruth';
  if scaling_choice==1
    x = x;
  end
  if scaling_choice==2
      x = x*255;
      x = (x-mean(x))./std(x);
  end
  if scaling_choice==3
      x = x*255;
      x = x./vecnorm(x);
  end
  % ESC parameters
  % m = number of atoms
  % k = number of clusters
  % t = number of nearest neighbours
  lambda = 200; m = 500; k = 5;
  % Call ESC
  groups = exemplar_subspace_clustering(x, k, m, lambda,3);
  % Accuracy
  groups = bestMap(gtruth, groups);
  accr  = sum(gtruth(:) == groups(:)) / length(gtruth)
  
end
 % ------------------------------------------------------------------------
 % Two moons dataset
 if data_choice ==4
     % Load two moons data
     load("./datasets/two_moons_10000.mat")
     % Specify number of data points
     num_data = 5000;
     % Prepare data matrix and labels
     x = double(data);
     x = x(1:num_data,:);
     x = x';
     labels_true = labels(1:num_data);
     labels_true = labels_true+1;
     gtruth = double(labels_true);
     % 
     if scaling_choice==1
        x = x/max(max(x));
     end
     if scaling_choice==2
        x = (x-mean(x))./std(x);
     end
     if scaling_choice==3
        x = x./vecnorm(x);
     end
     % ESC parameters
     % m = number of atoms
     % k = number of clusters
     % t = number of nearest neighbours
     lambda = 200; m = 24; k = 2;
     % Call ESC
     groups = exemplar_subspace_clustering(x, k, m, lambda,48);
     % Accuracy
     groups = bestMap(gtruth', groups);
     accr  = sum(gtruth(:) == groups(:)) / length(gtruth)

 end








