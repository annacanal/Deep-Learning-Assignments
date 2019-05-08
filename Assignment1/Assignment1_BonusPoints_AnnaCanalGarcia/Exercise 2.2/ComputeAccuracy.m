function acc = ComputeAccuracy(X, y, W)
    % each column of X corresponds to an image and X has size dxn.
    % y is the vector of ground truth labels of length n.
    % acc is a scalar value containing the accuracy.

	s = EvaluateClassifier(X, W);
    [argvalue, argmax]  = max(s); % argmax contains the class predictions
    correct = length(find(y - argmax == 0));
    acc = correct/length(y);
end
