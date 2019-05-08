function P = Evaluate(X, W, MF1,MF2)
    % Apply 1st convolutional layer
    X1 = max(MF1*X,0);
    % Apply 2nd convolutional layer
    X2 = max(MF2*X1,0);
    % Apply fully connected matrix
    Sbatch = W(:,1:size(X2,1))*X2;
    denom = repmat(sum(exp(Sbatch), 1), size(W, 1), 1);
    P = exp(Sbatch)./denom;
end