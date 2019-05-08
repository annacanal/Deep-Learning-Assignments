
%% Exercise 2.1: 1st change: decay learning rate after each epoch: 0.9
addpath Datasets/cifar-10-batches-mat/;
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');


X_validation = X_validation(:, 1:1000);
Y_validation = Y_validation(:, 1:1000);
y_validation = y_validation(:, 1:1000);

% bias trick
X_train = [X_train; ones(1, size(X_train, 2))];
X_validation = [X_validation; ones(1, size(X_validation, 2))];
X_test = [X_test; ones(1, size(X_test, 2))];

% initialize the parameters of the model W and b 
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.01;
W = mean + randn(K, d)*std;
b = mean + randn(K, 1)*std;
lambda=1.0;
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 60;
J_train = zeros(1, GDparams.n_epochs);
J_validation = zeros(1, GDparams.n_epochs);


for i=1: GDparams.n_epochs 
    J_train(i) = ComputeCost(X_train, Y_train, W, lambda);
    J_validation(i) = ComputeCost(X_validation, Y_validation, W, lambda); 
    Wstar = MiniBatchGD(X_train, Y_train, GDparams, W, lambda);
    W=Wstar;
    GDparams.eta = GDparams.eta*0.9; %decay eta 0.9 after each epoch
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
acc_train = ComputeAccuracy(X_train, y_train, W);
disp(['Training Accuracy:' num2str(acc_train*100) '%'])
acc_test = ComputeAccuracy(X_test, y_test, W);
disp(['Test Accuracy:' num2str(acc_test*100) '%'])

%visualize the weight matrix W as an image and see what class template the
%network has learnt
% K = 10;
% for i = 1 : K
%     im = reshape(W(i, :), 32, 32, 3);
%     s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
%     s_im{i} = permute(s_im{i}, [2, 1, 3]);
% end
% figure()
% montage(s_im, 'size', [1, K])
