function [grad_W, grad_b] = ComputeGradients(X, Y, P, h, W, b, lambda)
    % each column of X corresponds to an image and it has size dxn.
    % Y (Kxn) is the one-hot ground truth label for the corresponding column of X.
    % P has size Kxn.
    % grad_W is the gradient matrix of the cost J relative to W and has size Kxd.
    % grad_b is the gradient vector of the cost J relative to b and has size Kx1.

    W1 = cell2mat(W(1));
    b1 = cell2mat(b(1));
    W2 = cell2mat(W(2));
    b2 = cell2mat(b(2));   
    grad_W1 = zeros(size(W1));
    grad_W2 = zeros(size(W2));
    grad_b1 = zeros(size(b1));
    grad_b2 = zeros(size(b2));

    for i = 1 : size(X, 2)
        P_i = P(:, i);
        h_i = h(:, i);
        Y_i = Y(:, i);
        X_i = X(:, i);        
        g = -(Y_i-P_i)';
        grad_b2 = grad_b2 + g';
        grad_W2 = grad_W2 + g'*h_i';
        h_i(find(h_i > 0)) = 1;
        g = g*W2*diag(h_i);
        grad_b1 = grad_b1 + g';
        grad_W1 = grad_W1 + g'*X_i';   
    end

    % divide grad by the number of entries in D and apply regularization
    grad_b1 = grad_b1/size(X, 2);
    grad_W1 = grad_W1/size(X, 2) + 2*lambda*W1;
    grad_b2 = grad_b2/size(X, 2);
    grad_W2 = grad_W2/size(X, 2) + 2*lambda*W2;
    grad_W = {grad_W1, grad_W2}; 
    grad_b = {grad_b1, grad_b2};
end