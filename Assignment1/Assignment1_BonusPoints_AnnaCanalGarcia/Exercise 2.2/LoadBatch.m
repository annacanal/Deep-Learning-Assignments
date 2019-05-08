function [X, Y, y] = LoadBatch(filename)
% a function that reads in the data from a CIFAR-10 batch file 
% and returns the image and label data in separate files.
% X has size dxN
% Y has size KxN and contains the one-hot representation of the label for each image.
% y has length N containing the label for each image.

    A = load(filename);
    X = double(A.data')/255; % convert image from 0-256 to 0-1
    y = double(A.labels') + 1; % instead of classes from 0-9, classes from 1-10
    ind = y;
    vec = ind2vec(ind);
    Y =full(vec); % one-hot representation
    
end