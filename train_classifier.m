close all;
clear;
clc;

%% TODO
% Load training and test images and labels into image datastore
%   setup augmentation on datastores using MATLAB tools
%   split training datastore into train and validation
% Define neural network layers as described in pdf
%   improve upon pdf network (LSTM? more conv layers?)

% Training images that are used to train the network
% These images will be split into a two separate train and validation sets
imdsTrain = imageDatastore('TRAINING/*.png', 'ReadFcn', @imr2d);
Training_labels = load('Training_labels.mat');
%imdsTrain.Labels = categorical(cellstr(num2str(Training_labels.labels)));
imdsTrain.Labels = categorical(sort(Training_labels.labels));

% Test images play no part in training the network
% they are used only after training to test the performance of the network
imdsTest = imageDatastore('TESTING/*.png', 'ReadFcn', @imr2d);
Testing_labels = load('Testing_labels.mat');
%imdsTest.Labels = categorical(cellstr(num2str(Testing_labels.labels)));
imdsTest.Labels = categorical(sort(Testing_labels.labels));

% Split training datastore into two non-overlapping train and validation
% sets. The validation set is used by MATLAB's trainNetwork function
[imdsTrain, imdsVal] = splitEachLabel(imdsTrain, 0.8, 0.2);  

% "help <layername>" for more information on parameters
layers = [
    % First feature map is an image
	imageInputLayer([28 28 1])
    
    %Depth 1
    % convolves previous feature map by a 3x3 filter
    % learn values for num_channels=64 filters
    % https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % pool size [2,2], stride size [2,2]
    % dropoutLayer() %OPTIONAL
    
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2) % pool size [2,2], stride size [2,2]
    % dropoutLayer() % OPTIONAL
    
    fullyConnectedLayer(1024)
    reluLayer
    % dropoutLayer() % OPTIONAL
    
    fullyConnectedLayer(512)
    reluLayer
    % dropoutLayer() % OPTIONAL
    
    fullyConnectedLayer(128)
    reluLayer
    % dropoutLayer() % OPTIONAL
    
   
    % The output of the previous layer will have a different response for
    % different classes of inputs. Here, we classify these reponses
    
    
    %% OUTPUT LAYERS
    % Narrow down previous feature map to a vector as big as the number of
    % classes
    fullyConnectedLayer(10)
    %Normalises previous feature map from real numbers to probability
    %distributions in interval [0,1]
    softmaxLayer();
    %Computes cross-entropy loss of all prob distributions
    classificationLayer();
    
];

options = trainingOptions('adam', ...
    'MaxEpochs',15,...
    'ValidationData',imdsVal, ...
    'ValidationFreq',10, ...
    'InitialLearnRate',5e-5, ... 
    'ExecutionEnvironment', 'cpu', ...%'multi-gpu', ... % set to 'cpu' or remove line
    'Plots','training-progress', ...
    'Shuffle', 'every-epoch',...%);%, ... 
    'MiniBatchSize', 64); 

net = trainNetwork(imdsTrain, layers, options);
save net

function im = imr2d(file)
    im = im2double(imread(file));
end