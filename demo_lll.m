% ----------------------------------------------------------------------------------
% This is a demo script that runs the "Landmark based LLE" algorithm that
% is produced in the paper below
% Vladymyrov, Max, and Miguel Á. Carreira-Perpinán. "Locally linear landmarks 
% for large-scale manifold learning." Machine Learning and Knowledge Discovery in Databases: 
% European Conference, ECML PKDD 2013, Prague, Czech Republic, September 23-27, 2013, 
% Proceedings, Part III 13. Springer Berlin Heidelberg, 2013.
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
% Add path to different necessary folders
addpath('./lll_esc_codes/')
addpath(genpath('./SMCE'))
% Choose data and pre-processing
data_choice = 1;
% We consider three pre-processing techniques
% (1) Scaling to [0,1]
% (2) Standardizing i.e. N(0,1)
% (3) Normalizing each column of the data matrix by its l2 norm
scaling_choice = 1 ;
% ------------------------------------------------------------------------------------
% Yale Face Dataset
if data_choice == 1
    [x,gtruth]=load_yale;
    if scaling_choice==1
        x = x/max(max(x));
    end
    if scaling_choice==2
        x = (x-mean(x))./std(x);
    end
    if scaling_choice==3
        x = x./vecnorm(x);
    end
    % Call landmark LLE
    misrate= zeros(5,5);
    for i = 1:5
        [Y0,lx] = lmarks(x',64);		
        Z = lweights(x',Y0,64);		
        Z = full(Z);
        % clustering
        for j = 1:5
            labels = spectralcluster(Z',2,'LaplacianNormalization','symmetric','KernelScale','auto');
            misrate(i,j) = missclassGroups(labels,gtruth') ./ length(gtruth);
        end
    end
 end
% ----------------------------------------------------------------------------------
% Salinas A dataset
if data_choice == 2
    [x,gtruth]=load_salinas;
    if scaling_choice==1
        x = x/max(max(x));
    end
    if scaling_choice==2
        x = (x-mean(x))./std(x);
    end
    if scaling_choice==3
        x = x./vecnorm(x);
    end
    % Call landmark LLE
    misrate= zeros(5,5);
    for i = 1:5
        [Y0,lx] = lmarks(x',600);		
        Z = lweights(x',Y0,600);		
        Z = full(Z);
        % clustering
        for j = 1:5
            labels = spectralcluster(Z',6,'LaplacianNormalization','symmetric','KernelScale','auto');
            misrate(i,j) = missclassGroups(labels,gtruth') ./ length(gtruth)
        end
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
  % Call landmark LLE
    misrate= zeros(5,5);
    for i = 1:5
        d = 5;
        K = 50;
        [W,~] = gaussaff(x',{'k',K});
    	[X,Y0,X0,lx,Z] = lll(x',W,d,800);
        % clustering
        for j = 1:5
            labels = kmeans(X, 5, 'EmptyAction', 'singleton','Replicates',20,'MaxIter',1000);
            misrate(i,j) = missclassGroups(labels,gtruth') ./ length(gtruth);
        end
    end
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
    % Call landmark LLE
    misrate= zeros(5,5);
    for i = 1:5
        [Y0,lx] = lmarks(x',24);
        Z = lweights(x',Y0,24);	
        Z = full(Z);
        % clustering
        for j = 1:5
            labels = spectralcluster(Z',2,'LaplacianNormalization','symmetric','KernelScale','auto');
            misrate(i,j) = missclassGroups(labels,gtruth') ./ length(gtruth);
        end
    end
 end








