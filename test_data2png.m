%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Read and display the MNIST test images and labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
close all;
clear all;
clc;
rng('default');
% Load the training images and the labels
images = loadMNISTImages('t10k-images-idx3-ubyte');
labels = loadMNISTLabels('t10k-labels-idx1-ubyte');

% Select a random subset of the data
N = 400; 
S = randsample(size(labels,1),N);
images = images(:,S);
labels = labels(S);

image_size = sqrt(size( images, 1));
Testing='TESTING';
mkdir(Testing);
for i =1:length(labels)
    img_matrix = reshape( images(:, i), [ image_size ,image_size ]);
    img_matrix = floor( img_matrix *255);
    img_matrix =uint8(img_matrix);
    p = labels(i,:);
    baseFileName = sprintf('%d%d.png',p,i); 
    fullFileName = fullfile(Testing, baseFileName); 
    imwrite(img_matrix, fullFileName);
end

save Testing_labels labels
