function [RNN, M] = backward(RNN, X, Y, a, h, p, n, m, eta, M)
   % compute gradients : grads.c, grads.V, grads.b, grads.W and grads.U 
    grads = ComputeGradients(RNN, X, Y, a, h, p, n, m);
    eps = 1e-8;
    % clip gradients to avoid exploding gradient
    for f=fieldnames(grads)'
        grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
    
    for f = fieldnames(RNN)'
        % AdaGrad
        M.(f{1}) = M.(f{1}) + grads.(f{1}).^2;
        RNN.(f{1}) = RNN.(f{1}) - eta*(grads.(f{1})./(M.(f{1}) + eps).^(0.5));
        %RNN.(f{1}) = RNN.(f{1}) - eta*(grads.(f{1}));
    end

end