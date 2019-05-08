function grad_W= ComputeGradients(X, Y, s, W, lambda)
    % each column of X corresponds to an image and it has size dxn.
    % Y (Kxn) is the one-hot ground truth label for the corresponding column of X.
    % P has size Kxn.
    % grad_W is the gradient matrix of the cost J relative to W and has size Kxd.
    % grad_b is the gradient vector of the cost J relative to b and has size Kx1.

    grad_W = zeros(size(W));
	sy = repmat(sum(s.*Y), size(s, 1), 1);
	margin = s - sy + 1;
	flag = zeros(size(s));
	flag(find(margin > 0)) = 1;
	flag(find(Y == 1)) = -1;

	for i = 1 : size(X, 2)
		Xi = X(:, i);
		fi = flag(:, i);
		gi = repmat(Xi', size(W, 1), 1);
		gi(find(fi == 0), :) = 0;
		gi(find(fi == -1), :) = -length(find(fi == 1))*gi(find(fi == -1), :);
		grad_W = grad_W + gi;
	end
	grad_W = grad_W/size(X, 2) + 2*lambda*W ;

end