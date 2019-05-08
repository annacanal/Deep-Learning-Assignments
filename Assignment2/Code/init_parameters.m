function [W, b] = init_parameters(m, K, d, std, mean)
    W1 = mean + randn(m, d)*std;
    W2 = mean +  randn(K, m)*std;
    W= {W1, W2};
    b1 = zeros(m, 1);
    b2 = zeros(K, 1);
    b= {b1, b2};

end