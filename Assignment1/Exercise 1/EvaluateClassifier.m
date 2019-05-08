function P = EvaluateClassifier(X, W, b)
% evaluates the network function on multiple images and returns the results.

% each column of X corresponds to an image and it has size dxn.
% W and b are the parameters of the network.
% each column of P contains the probability for each label for the image in the corresponding column of X. 
% P has size Kxn.
    b = repmat(b, 1, size(X, 2));
    s = W*X + b;
    denom = repmat(sum(exp(s), 1), size(W, 1), 1);
    P = exp(s)./denom;

end
