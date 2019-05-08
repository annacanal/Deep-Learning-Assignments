
%% Exercise 2.1: 1st change: decay learning rate after each epoch: 0.9
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
lambda=0.0;
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

%% Exercise 2.1: 2nd change: longer training data, less validation
addpath Datasets/cifar-10-batches-mat/;

[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
X_validation = X_train2(:, 1:1000);
Y_validation = Y_train2(:, 1:1000);
y_validation = y_train2(:, 1:1000);
X_train2 = X_train2(:, 1001:10000);
Y_train2 = Y_train2(:, 1001:10000);
y_train2 = y_train2(:, 1001:10000);
% X_train = [X_train1, X_train2, X_train3, X_train4, X_train5];
% Y_train = [Y_train1, Y_train2, Y_train3, Y_train4, Y_train5];
% y_train = [y_train1, y_train2, y_train3, y_train4, y_train5];
X_train = [X_train1, X_train3];
Y_train = [Y_train1, Y_train3];
y_train = [y_train1, y_train3];

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
    GDparams.eta = GDparams.eta*0.9; %decay eta 0.9 after each epoch
end

% Accuracy of the network
acc_train = ComputeAccuracy(X_train, y_train, W, b);
disp(['Training Accuracy:' num2str(acc_train*100) '%'])
acc_test = ComputeAccuracy(X_test, y_test, W, b);
disp(['Test Accuracy:' num2str(acc_test*100) '%'])
    
%% Exercise 2.3: 3rd change: longer training time
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
lambda=0.0;
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 50;
J_train = zeros(1, GDparams.n_epochs);
J_validation = zeros(1, GDparams.n_epochs);

for i=1: GDparams.n_epochs 
    J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
    J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
    [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
    W=Wstar;
    b=bstar;
   % GDparams.eta = GDparams.eta*0.9; %decay eta 0.9 after each epoch
end

% Accuracy of the network
acc_train = ComputeAccuracy(X_train, y_train, W, b);
disp(['Training Accuracy:' num2str(acc_train*100) '%'])
acc_test = ComputeAccuracy(X_test, y_test, W, b);
disp(['Test Accuracy:' num2str(acc_test*100) '%'])


%% Exercise 2.4: Grid Search
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
n_lambda=10;
n_eta=10;
lambda=linspace(0,1,n_lambda);
GDparams.n_batch = 100;
GDparams.eta = linspace(0.001,0.1,n_eta);
GDparams.n_epochs = 40;
J_train = zeros(1, GDparams.n_epochs);
J_validation = zeros(1, GDparams.n_epochs);

lambda_max = -1e+10;
eta_max = -1e+10;
accuracy_max = -1e+10;


for j=1:n_lambda
    for x=1:n_eta
        for i=1: GDparams.n_epochs 
            J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda(j));
            J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda(j)); 
            [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda(j), GDparams.eta(x));
            W=Wstar;
            b=bstar;
            % GDparams.eta = GDparams.eta*0.9; %decay eta 0.9 after each epoch
        end
        % Accuracy of the network
        acc_test = ComputeAccuracy(X_test, y_test, W, b);
        disp(['Test Accuracy:' num2str(acc_test*100) '%'])
        if acc_test >= accuracy_max 
            lambda_max = lambda(j);
            eta_max = GDparams.eta(x);
            accuracy_max = acc_test;
        end
        disp(['Test Accuracy Max with lambda= ' num2str(lambda_max) ' and eta= ' num2str(eta_max)])
    end
end


%% Exercise 2.1: final 
addpath Datasets/cifar-10-batches-mat/;
% [X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
% [X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
% [X_test, Y_test, y_test]= LoadBatch('test_batch.mat');
% 
% % less validation
% X_validation = X_validation(:, 1:1000);
% Y_validation = Y_validation(:, 1:1000);
% y_validation = y_validation(:, 1:1000);

[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
X_validation = X_train2(:, 1:1000);
Y_validation = Y_train2(:, 1:1000);
y_validation = y_train2(:, 1:1000);
X_train2 = X_train2(:, 1001:10000);
Y_train2 = Y_train2(:, 1001:10000);
y_train2 = y_train2(:, 1001:10000);
X_train = [X_train1, X_train3];
Y_train = [Y_train1, Y_train3];
y_train = [y_train1, y_train3];

[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% initialize the parameters of the model W and b 
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.01;
W = mean + randn(K, d)*std;
b = mean + randn(K, 1)*std;
lambda=0.0;
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 60;
J_train = zeros(1, GDparams.n_epochs);
J_validation = zeros(1, GDparams.n_epochs);

for i=1: GDparams.n_epochs 
    J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
    J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
    [Wstar, bstar] = MiniBatchGD(X_train, Y_train, GDparams, W, b, lambda);
    W=Wstar;
    b=bstar;
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
acc_train = ComputeAccuracy(X_train, y_train, W, b);
disp(['Training Accuracy:' num2str(acc_train*100) '%'])
acc_test = ComputeAccuracy(X_test, y_test, W, b);
disp(['Test Accuracy:' num2str(acc_test*100) '%'])

