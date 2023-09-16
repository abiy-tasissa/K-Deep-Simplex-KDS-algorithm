% ----------------------------------------------------------------------------------
% This is a demo script that runs SMCE experiments from the paper
% SMCE is an abbreviation for sparse manifold clustering and embedding.
% The algorithm was proposed in the paper referenced below:
% Ref: Elhamifar, E., & Vidal, R. (2011), Sparse manifold clustering and embedding. 
%      Advances in neural information processing systems, 24, 55-63.
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
% Here, we only consider the 5 digits 0,3,4,6,7. Reason for choice (a) This
% choice was shown in the SMCE paper to which we are comparing our
% algorithm to (b) SMCE is prohibitively slow to not being able to run
% on standard computing for full MNIST (unless the K_max parameter and
% lambda are chosen suitably)
% --------------------------------------------------------------------------------
% choice = 4 is a synthetic dataset of two moons. This dataset can be
% generated using sklearn in PYTHON
% Reference:
% https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html
% This data is included along with this script
% ----------------------------------------------------------------------------------
% Add path to SMCE folder
addpath(genpath('./SMCE'))
% Choose data and pre-processing
data_choice = 1;
% We consider three pre-processing techniques
% (1) Scaling to [0,1]
% (2) Standardizing i.e. N(0,1)
% (3) Normalizing each column of the data matrix by its l2 norm
scaling_choice = 3 ;
% ------------------------------------------------------------------------------------
% Yale Face Dataset
if data_choice == 1
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
    % SMCE parameters
    lambda = 200; KMax = 64; dim = 2;
    n = length(unique(labels));
    verbose = true;
    gtruth = labels';
    % Call SMCE
   [Yc,Yj,clusters,missrate] = smce(x,lambda,KMax,dim,n,gtruth,verbose);
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
    % SMCE parameters
    lambda = 1; KMax = 600; dim = 2;
    n = length(unique(labels));
    verbose = true;
    gtruth = labels';
    % Call SMCE
    [Yc,Yj,clusters,missrate] = smce(x,lambda,KMax,dim,n,gtruth,verbose);
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
  gtruth = gtruth(1:num_data)';
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
  % SMCE parameters
  lambda = 1; KMax = 500; dim = 2;
  n = length(unique(labels));
  verbose = true;
  tic
  [missrate] = smce_mnist(x,lambda,KMax,dim,n,gtruth,verbose);
  toc

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
     % SMCE parameters
     lambda = 200; KMax = 24; dim = 2;
     n = 2;
     verbose = true;
     % Call SMCE
     tic;
     [Yc,Yj,clusters,missrate] = smce(x,lambda,KMax,dim,n,gtruth,verbose);
 end








