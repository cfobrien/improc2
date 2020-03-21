%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Read and display the MNIST train images and the labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear all;
rng('default');
% Load the training images and the labels
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');

% Select a random subset of the data
N = 2000; 
S = randsample(size(labels,1),N);
images = images(:,S);
labels = labels(S);

image_size = sqrt(size( images, 1));
Training='TRAINING';
mkdir(Training);
for i =1:length(labels)
    img_matrix = reshape( images(:, i), [ image_size ,image_size ]);
    img_matrix = floor( img_matrix *255);
    img_matrix =uint8(img_matrix);
    p = labels(i,:);
    baseFileName = sprintf('%d%d.png',p,i); 
    fullFileName = fullfile(Training, baseFileName); 
    imwrite(img_matrix, fullFileName);
end

save Training_labels labels