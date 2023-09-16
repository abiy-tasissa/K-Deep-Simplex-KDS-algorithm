% -------------------------------------------------------------------------
% This script loads the Yale Face dataset
% Download the dataset from the link below
% http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html
% to a folder of your choice. 
% To preserve anonymity, here is an example of how to run the code 
% If john_Doe is the user name and the folder is in Downloads, 
% the path can be specified as
% P = '/home/john_doe/Downloads/CroppedYale/yaleB05'
% -------------------------------------------------------------------------
function [x,labels] = load_yale
    % Specify path
    Path = "./data/yale";
    % Specify how many faces. We consider 2,3 or 4 faces. 
    num_faces = 2;
    % Load Face 1
    P = Path+'/yaleB05';
    D = dir(fullfile(P,'*.pgm'));
    C = cell(size(D));
    for k = 1:numel(D)
       C{k} = imread(fullfile(P,D(k).name)); 
    end	
    % Form the data matrix
    faces1 = zeros(192*168,64);
    for i = 1:64
        faces1(:,i) = reshape(C{i,1},192*168,1);
    end
    % Load Face 2
    P = Path+'/yaleB09';
    D = dir(fullfile(P,'*.pgm'));
    C = cell(size(D));
    for k = 1:numel(D)
       C{k} = imread(fullfile(P,D(k).name)); 
    end	
    % Form the data matrix
    faces2 = zeros(192*168,64);
    for i = 1:64
        faces2(:,i) = reshape(C{i,1},192*168,1);
    end
    % Load Face 3
    P = Path+'/yaleB03';
    D = dir(fullfile(P,'*.pgm'));
    C = cell(size(D));
    for k = 1:numel(D)
       C{k} = imread(fullfile(P,D(k).name)); 
    end	
    % Form the data matrix
    faces3 = zeros(192*168,64);
    for i = 1:64
        faces3(:,i) = reshape(C{i,1},192*168,1);
    end
    % Load Face 4
    P = Path+'/yaleB04';
    D = dir(fullfile(P,'*.pgm'));
    C = cell(size(D));
    for k = 1:numel(D)
       C{k} = imread(fullfile(P,D(k).name)); 
    end	
    % Form the data matrix
    faces4 = zeros(192*168,64);
    for i = 1:64
        faces4(:,i) = reshape(C{i,1},192*168,1);
    end
    % Prepare data matrix and labels
    if num_faces == 2
        x = [faces1 faces2  ];
        labels = ones(128,1);
        labels(1:64) = 1;
        labels(65:128)= 2;
    end
    if num_faces == 3
        x = [faces1 faces2 faces3  ];
        labels = ones(128,1);
        labels(1:64) = 1;
        labels(65:128)= 2;
        labels(129:192)= 3;
    end
    if num_faces == 4
        x = [faces1 faces2 faces3 faces4  ];
        labels = ones(128,1);
        labels(1:64) = 1;
        labels(65:128)= 2;
        labels(129:192)= 3;
        labels(193:256)=4;
    end
    
    
end
