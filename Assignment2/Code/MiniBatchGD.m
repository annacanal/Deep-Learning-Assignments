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
        [P,h] = EvaluateClassifier(Xbatch, W, b);
        [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, h, W, b, lambda);
        % update weights and bias
        W{1} = W{1} - eta*grad_W{1};
        b{1}  = b{1} - eta*grad_b{1};
        W{2} = W{2} - eta*grad_W{2};
        b{2}  = b{2} - eta*grad_b{2};        
    end
Wstar = W;
bstar = b;
end