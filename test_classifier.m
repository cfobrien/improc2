close all;
clear;
clc;

% Test images play no part in training the network
% they are used only after training to test the performance of the network
imdsTest = imageDatastore('TESTING/*.png', 'ReadFcn', @imr2d);
Testing_labels = load('Testing_labels.mat');
imdsTest.Labels = categorical(cellstr(num2str(Testing_labels.labels)));
imdsTest = shuffle(imdsTest);

load net

num_test = 100;
rand_indices = randi(num_test, numel(imdsTest.Files), 1); %indices associated with num_test random test images
actual = zeros(num_test, 1);
predicted = zeros(num_test, 1);
for i = 1 : num_test
    im = readimage(imdsTest, i);
    %compute actual and predicted classes of num_test random test images
    [argval, argmax] = max(predict(net, im));
    predicted(i) = argmax;
    actual(i) = imdsTest.Labels(i);
end

%compute and display confusion matrix
C = confusionmat(actual, predicted);
confusionchart(C);

function im = imr2d(file)
    im = im2double(imread(file));
end