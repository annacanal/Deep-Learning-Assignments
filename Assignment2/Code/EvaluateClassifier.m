function [P,h] = EvaluateClassifier(X, W, b)
% evaluates the network function on multiple images and returns the results.
% each column of X corresponds to an image and it has size dxn.
% W and b are the parameters of the network.
% each column of P contains the probability for each label for the image in the corresponding column of X. 
% P has size Kxn.
    W1 = cell2mat(W(1));
    b1 = cell2mat(b(1));
    W2 = cell2mat(W(2));
    b2 = cell2mat(b(2));   
    b1 = repmat(b1, 1, size(X, 2));   
    s1 = W1*X + b1;
    h = max(0,s1);
    b2 = repmat(b2, 1, size(h, 2));
    s2 = W2*h + b2;
    denom = repmat(sum(exp(s2), 1), size(W2, 1), 1);
    P = exp(s2)./denom;
end
