%% Functions

%% Exercise 1:
addpath Datasets/cifar-10-batches-mat/;

[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% initialize the parameters of the model W and b 
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.01;
W = mean + randn(K, d)*std;
b = mean + randn(K, 1)*std;
% P = EvaluateClassifier(X_train(:, 1:100), W, b);
lambda=0.1;
% J = ComputeCost(X_train(:, 1:100), Y_train(:, 1:100), W, b, lambda);
% check if analytical gradient is correct
batch_size = 50;
eps =1e-10;

[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(:, 1 : batch_size), Y_train(:, 1 : batch_size), W, b, lambda, 1e-6);
P = EvaluateClassifier(X_train(:, 1 : batch_size),  W, b);
[grad_W, grad_b] = ComputeGradients(X_train(:, 1 : batch_size), Y_train(:, 1 : batch_size), P,  W, lambda);
%check gradients
gradcheck_b = max(abs(ngrad_b - grad_b)./max(eps, abs(ngrad_b) + abs(grad_b)));
gradcheck_W = max(max(abs(ngrad_W - grad_W)./max(eps, abs(ngrad_W) + abs(grad_W))));

if gradcheck_b <= 1e-6 
    fprintf("Correct grad_b");
else
    fprintf("Incorrect grad_b");
end
if gradcheck_W <= 1e-6 
    fprintf("Correct grad_W");
else
    fprintf("Incorrect grad_W");
end

%% Exercise 1: Training a multi-linear classiffier
addpath Datasets/cifar-10-batches-mat/;
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% initialize the parameters of the model W and b 
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.01;
W = mean + randn(K, d)*std;
b = mean + randn(K, 1)*std;
lambda=0.1;
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 40;
J_train = zeros(1, GDparams.n_epochs);
J_validation = zeros(1, GDparams.n_epochs);

for i=1: GDparams.n_epochs 
    J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
    J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
    [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
    W=Wstar;
    b=bstar;
end

%plot cost score
figure()
plot(1 : GDparams.n_epochs, J_train, 'g')
hold on
plot(1 : GDparams.n_epochs, J_validation, 'r')
hold off
xlabel('epochs');
ylabel('loss');
legend('Training loss', 'Validation loss');

% Accuracy of the network
acc_train = ComputeAccuracy(X_train, y_train, W, b);
disp(['Training Accuracy:' num2str(acc_train*100) '%'])
acc_test = ComputeAccuracy(X_test, y_test, W, b);
disp(['Test Accuracy:' num2str(acc_test*100) '%'])

%visualize the weight matrix W as an image and see what class template the
%network has learnt
K = 10;
for i = 1 : K
    im = reshape(W(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end
figure()
montage(s_im, 'size', [1, K])

    
