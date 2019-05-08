function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, lambda)
    % each column of X corresponds to an image and it has size dxn.
    % Y (Kxn) is the one-hot ground truth label for the corresponding column of X.
    % P has size Kxn.
    % grad_W is the gradient matrix of the cost J relative to W and has size Kxd.
    % grad_b is the gradient vector of the cost J relative to b and has size Kx1.

    grad_W = zeros(size(W));
    grad_b = zeros(size(W, 1), 1);

    for i = 1 : size(X, 2)
        P_i = P(:, i);
        Y_i = Y(:, i);
        X_i = X(:, i);        
        %g = -Y_i'/(Y_i'*P_i)*(diag(P_i) - P_i*P_i'); 
        % simplification g = -Y_i'/(Y_i'*P_i)*(diag(P_i) - P_i*P_i') = (Y_i-P_i)'
        g = -(Y_i-P_i)';
        grad_b = grad_b + g';
        grad_W = grad_W + g'*X_i';
 
    end
    % divide grad by th enumber of entries in D
    grad_b = grad_b/size(X, 2);
    grad_W = grad_W/size(X, 2) + 2*lambda*W ;

end