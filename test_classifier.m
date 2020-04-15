close all;
clear;
clc;

% Test images play no part in training the network
% they are used only after training to test the performance of the network
imdsTest = imageDatastore('TESTING/*.png', 'ReadFcn', @imr2d);
Testing_labels = load('Testing_labels.mat');
imdsTest.Labels = categorical(cellstr(num2str(Testing_labels.labels)));

load net

fprintf("Press any key for next image or Ctrl+C to exit\n");
fprintf("Actual class  |  Predicted Class\n");
fprintf("================================\n");

pred = [];
for i = 1 : numel(imdsTest.Files)
        
    im = readimage(imdsTest, i);
    imshow(im);
    
    %[argval, argmax] = max(predict(net, im));
    class = classify(net, im);
    pred = [pred class];
    fprintf("       %d               %d\n", imdsTest.Labels(i), class);
    
    %pause;
end
cm = confusionchart(imdsTest.Labels, pred');

function im = imr2d(file)
    im = im2double(imread(file));
end