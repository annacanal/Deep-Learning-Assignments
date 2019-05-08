function [Wstar, bstar] = MiniBatchGD(X, Y, GDparams, W, b, lambda)

    n_batch = GDparams.n_batch;
    eta = GDparams.eta;
    N = size(X, 2);
    
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        Xbatch = X(:, inds);
        Ybatch = Y(:, inds);  
        % compute gradients for each mini-batch
        P = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, lambda);
        % update weights and bias
        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end
Wstar = W;
bstar = b;
end