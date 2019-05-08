function J = ComputeCost(X, Y, W, b, lambda)
    % each column of X corresponds to an image and X has size dxn.
    % each column of Y (Kxn) is the one-hot ground truth label for the corresponding column of X or Y is the (1xn) vector of ground truth labels.
    % J is a scalar corresponding to the sum of the loss of the network's predictions for the images in X relative to the ground truth labels and the regularization term on W.
    W1 = cell2mat(W(1));
    W2 = cell2mat(W(2));
    [P,h] = EvaluateClassifier(X, W, b);
    lcross = -log(Y'*P);
    regularization = lambda* ( sum(sum(W1.^2)) +  sum(sum(W2.^2)));
    J = sum(diag(lcross))/size(X, 2) + regularization;    
end

