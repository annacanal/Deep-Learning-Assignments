function Wstar = MiniBatchGD(X, Y, GDparams, W, lambda)

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
        s = EvaluateClassifier(Xbatch, W);
        grad_W = ComputeGradients(Xbatch, Ybatch, s, W, lambda);
        % update weights 
        W = W - eta*grad_W;
    end
	Wstar = W;
end