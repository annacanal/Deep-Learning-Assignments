function J = Compute_loss(X_batch, Ys_batch, ConvNet)
% X_batch: nlen * d x n
% Ys_batch: one hot encoding of the labels of each example
% MF1: MF matrix of each layer 1
% MF2: MF matrix of each layer 2
% W: weigth matrix for the last fully connected layer

    MF1 = ConvNet.MF{1};
    MF2 = ConvNet.MF{2};
    W = ConvNet.W;
    P_batch = Evaluate(X_batch, W, MF1,MF2);
    lcross = -log(diag(Ys_batch'*P_batch)+ 1e-10);
    J = sum(lcross)/size(X_batch, 2) ;  
end