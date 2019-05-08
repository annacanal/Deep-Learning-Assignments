function acc = ComputeAccuracy(X, ys, ConvNet)
    % each column of X corresponds to an image and X has size dxn.
    % y is the vector of ground truth labels of length n.
    % acc is a scalar value containing the accuracy.

    %Creac noves MF1 i MF2 amb tota la X
    MF1 = ConvNet.MF{1};
    MF2 = ConvNet.MF{2};
    W = ConvNet.W;
    P = Evaluate(X, W, MF1,MF2);
    [argvalue, argmax]  = max(P);% argmax contains the class predictions
    correct = length(find(ys - argmax == 0));
    acc = correct/length(ys);
end