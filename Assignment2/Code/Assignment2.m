
%% Exercise 1 and 2:
addpath Datasets/cifar-10-batches-mat/;

[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% Zero mean data pre-processing
%X = logical(X_train);
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_validation = X_validation - repmat(mean_X, [1, size(X_validation, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the parameters of the model W1, W2, b1 and b 2
m = 50; %hidden nodes
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.001;
[W,b]= init_parameters(m,K,d,std,mean);

lambda=0;
batch_size = 100;
eps =1e-4;
% numerical gradients
[ngrad_b, ngrad_W] = ComputeGradsNumSlow(X_train(:, 1 : batch_size), Y_train(:, 1 : batch_size), W, b, lambda, 1e-5);
% analytical gradients
[P,h] = EvaluateClassifier(X_train(:, 1 : batch_size),  W, b);
[grad_W, grad_b] = ComputeGradients(X_train(:, 1 : batch_size), Y_train(:, 1 : batch_size), P, h, W, b, lambda);
%check gradients
gradcheck_b1 = sum(abs(ngrad_b{1} - grad_b{1})/max(eps, sum(abs(ngrad_b{1}) + abs(grad_b{1}))));
gradcheck_W1 = sum(sum(abs(ngrad_W{1} - grad_W{1})/max(eps, sum(sum(abs(ngrad_W{1}) + abs(grad_W{1}))))));
gradcheck_b2 = sum(abs(ngrad_b{2} - grad_b{2})/max(eps, sum(abs(ngrad_b{2}) + abs(grad_b{2}))));
gradcheck_W2 = sum(sum(abs(ngrad_W{2} - grad_W{2})/max(eps, sum(sum(abs(ngrad_W{2}) + abs(grad_W{2}))))));
gradcheck_W = [gradcheck_W1, gradcheck_W2];
gradcheck_b = [gradcheck_b1, gradcheck_b2];

if gradcheck_b1 <= 1e-6 
    fprintf("Correct grad_b1");
else
    fprintf("Incorrect grad_b1");
end
if gradcheck_W1 <= 1e-6 
    fprintf("Correct grad_W1");
else
    fprintf("Incorrect grad_W1");
end

if gradcheck_b2 <= 1e-6 
    fprintf("Correct grad_b2");
else
    fprintf("Incorrect grad_b2");
end
if gradcheck_W2 <= 1e-6 
    fprintf("Correct grad_W2");
else
    fprintf("Incorrect grad_W2");
end

%% Exercise 2
addpath Datasets/cifar-10-batches-mat/;

[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% Zero mean data pre-processing
%X = logical(X_train);
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_validation = X_validation - repmat(mean_X, [1, size(X_validation, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the parameters of the model W1, W2, b1 and b 2
m = 50; %hidden nodes
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.001;
[W,b]= init_parameters(m,K,d,std,mean);
lambda=0.0;
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 200;
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


%% Exercise 3
addpath Datasets/cifar-10-batches-mat/;

[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% Zero mean data pre-processing
%X = logical(X_train);
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_validation = X_validation - repmat(mean_X, [1, size(X_validation, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the parameters of the model W1, W2, b1 and b 2
m = 50; %hidden nodes
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.001;
%[W,b]= init_parameters(m,K,d,std,mean);
lambda=0.0;
GDparams.n_batch = 100;
GDparams.eta = 0.01;
GDparams.n_epochs = 100;
decay_rate= 0.95;


for rho=[0.5,0.9,0.99]
    GDparams.eta = 0.01;
    [W,b]= init_parameters(m,K,d,std,mean);
    J_train = zeros(1, GDparams.n_epochs);
    J_validation = zeros(1, GDparams.n_epochs);
    for i=1: GDparams.n_epochs 
        J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
        J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
        [Wstar, bstar] = MiniBatchGD_momentum(X_train, Y_train, GDparams, W, b, lambda, rho);
        W=Wstar;
        b=bstar;
        GDparams.eta=GDparams.eta*decay_rate;
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
    title(['Lambda = ',num2str(lambda),', learning rate = ',num2str(GDparams.eta),', decay rate = ',num2str(decay_rate),', rho = ',num2str(rho)]);
    
    % Accuracy of the network
    disp(['Network with rho = :' num2str(rho)])
    acc_train = ComputeAccuracy(X_train, y_train, W, b);
    disp(['Training Accuracy:' num2str(acc_train*100) '%'])
    acc_test = ComputeAccuracy(X_test, y_test, W, b);
    disp(['Test Accuracy:' num2str(acc_test*100) '%'])
end

%% Exercise 4: Coarse search
addpath Datasets/cifar-10-batches-mat/;

[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% Zero mean data pre-processing
%X = logical(X_train);
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_validation = X_validation - repmat(mean_X, [1, size(X_validation, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the parameters of the model W1, W2, b1 and b 2
m = 50; %hidden nodes
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.001;
GDparams.n_batch = 100;
GDparams.n_epochs = 10;
decay_rate= 0.95;
rho=0.9;
%lambda = 0.000001;
n_pairs= 70;
eta_max = 0.3;
eta_min = 0.001;
lambda_max = 0.1;
lambda_min = 1e-7;

eta_arr =[];
lambda_arr =[];
acc_val_arr=[];

for j = 1 : n_pairs
    % randomly generate eta and lambda
    e1 = log10(eta_min) + (log10(eta_max) - log10(eta_min))*rand(1, 1); 
    eta = 10^e1;
    e2 = log10(lambda_min) + (log10(lambda_max) - log10(lambda_min))*rand(1, 1);
    lambda = 10^e2;
    GDparams.eta = eta;
    
    [W,b]= init_parameters(m,K,d,std,mean);
    J_train = zeros(1, GDparams.n_epochs);
    J_validation = zeros(1, GDparams.n_epochs);
    for i=1: GDparams.n_epochs 
        J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
        J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
        [Wstar, bstar] = MiniBatchGD_momentum(X_train, Y_train, GDparams, W, b, lambda, rho);
        W=Wstar;
        b=bstar;
        GDparams.eta=GDparams.eta*decay_rate;
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
    title(['Lambda = ',num2str(lambda),' ,Learning rate = ',num2str(GDparams.eta)]);
   % title(['Lambda = ',num2str(lambda),', learning rate = ',num2str(GDparams.eta),', decay rate = ',num2str(decay_rate),', rho = ',num2str(rho)]);
   
    % Accuracy of the network
    disp(['Network with rho = :' num2str(rho)])
    acc_train = ComputeAccuracy(X_train, y_train, W, b);
    disp(['Training Accuracy:' num2str(acc_train*100) '%'])
    acc_val = ComputeAccuracy(X_validation, y_validation, W, b);
    disp(['Validation Accuracy:' num2str(acc_val*100) '%'])
    
    %save parameters and validation accuracy
    eta_arr = [eta_arr, eta];
    lambda_arr = [lambda_arr, lambda];
    acc_val_arr = [acc_val_arr, acc_val];
end
%Write file
filename1 = 'random_coarse_search.xlsx';
xlswrite(filename1, acc_val_arr,1, 'B2');
xlswrite(filename1, eta_arr,1, 'B3');
xlswrite(filename1, lambda_arr,1, 'B4');

%% Exercise 4: Fine search
addpath Datasets/cifar-10-batches-mat/;

[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_validation, Y_validation, y_validation] = LoadBatch('data_batch_2.mat');
[X_test, Y_test, y_test]= LoadBatch('test_batch.mat');

% Zero mean data pre-processing
%X = logical(X_train);
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_validation = X_validation - repmat(mean_X, [1, size(X_validation, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the parameters of the model W1, W2, b1 and b 2
m = 50; %hidden nodes
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.001;
GDparams.n_batch = 100;
GDparams.n_epochs = 15;
decay_rate= 0.95;
rho=0.9;
%lambda = 0.000001;
n_pairs= 50;
eta_max = 0.06;
eta_min = 0.01;
lambda_max = 0.01;
lambda_min = 1e-6;
% eta_max = 0.04;
% eta_min = 0.02;
% lambda_max = 0.01;
% lambda_min = 1e-4;

eta_arr =[];
lambda_arr =[];
acc_val_arr=[];

for j = 1 : n_pairs
    % randomly generate eta and lambda
    e1 = log10(eta_min) + (log10(eta_max) - log10(eta_min))*rand(1, 1); 
    eta = 10^e1;
    e2 = log10(lambda_min) + (log10(lambda_max) - log10(lambda_min))*rand(1, 1);
    lambda = 10^e2;
    GDparams.eta = eta;
    
    [W,b]= init_parameters(m,K,d,std,mean);
    J_train = zeros(1, GDparams.n_epochs);
    J_validation = zeros(1, GDparams.n_epochs);
    for i=1: GDparams.n_epochs 
        J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
        J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
        [Wstar, bstar] = MiniBatchGD_momentum(X_train, Y_train, GDparams, W, b, lambda, rho);
        W=Wstar;
        b=bstar;
        GDparams.eta=GDparams.eta*decay_rate;
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
    title(['Lambda = ',num2str(lambda),' ,Learning rate = ',num2str(eta)]);
   % title(['Lambda = ',num2str(lambda),', learning rate = ',num2str(GDparams.eta),', decay rate = ',num2str(decay_rate),', rho = ',num2str(rho)]);
   
    % Accuracy of the network
    disp(['Network setting = :' num2str(j)])
    acc_train = ComputeAccuracy(X_train, y_train, W, b);
    disp(['Training Accuracy:' num2str(acc_train*100) '%'])
    acc_val = ComputeAccuracy(X_validation, y_validation, W, b);
    disp(['Validation Accuracy:' num2str(acc_val*100) '%'])
    
    %save parameters and validation accuracy
    eta_arr = [eta_arr, eta];
    lambda_arr = [lambda_arr, lambda];
    acc_val_arr = [acc_val_arr, acc_val];
end
%Write file
filename2 = 'random_fine_search_2.xlsx';
xlswrite(filename2, acc_val_arr,1, 'B2');
xlswrite(filename2, eta_arr,1, 'B3');
xlswrite(filename2, lambda_arr,1, 'B4');

%% Exercise 4: Best hyper parameter
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

% Zero mean data pre-processing
%X = logical(X_train);
mean_X = mean(X_train, 2);
X_train = X_train - repmat(mean_X, [1, size(X_train, 2)]);
X_validation = X_validation - repmat(mean_X, [1, size(X_validation, 2)]);
X_test = X_test - repmat(mean_X, [1, size(X_test, 2)]);

% initialize the parameters of the model W1, W2, b1 and b 2
m = 50; %hidden nodes
K = size(Y_train, 1);
d = size(X_train, 1);
mean = 0;
std = 0.001;
lambda=0.00561 ;
GDparams.n_batch = 100;
eta=0.03546;
GDparams.eta = eta;
GDparams.n_epochs = 30;
decay_rate= 0.95;
rho=0.9;
[W,b]= init_parameters(m,K,d,std,mean);
J_train = zeros(1, GDparams.n_epochs);
J_validation = zeros(1, GDparams.n_epochs);
for i=1: GDparams.n_epochs 
    J_train(i) = ComputeCost(X_train, Y_train, W, b, lambda);
    J_validation(i) = ComputeCost(X_validation, Y_validation, W, b, lambda); 
    [Wstar, bstar] = MiniBatchGD_momentum(X_train, Y_train, GDparams, W, b, lambda, rho);
    W=Wstar;
    b=bstar;
    GDparams.eta=GDparams.eta*decay_rate;
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
title(['Lambda = ',num2str(lambda),', learning rate = ',num2str(eta),', decay rate = ',num2str(decay_rate),', rho = ',num2str(rho)]);

% Accuracy of the network
acc_train = ComputeAccuracy(X_train, y_train, W, b);
disp(['Training Accuracy:' num2str(acc_train*100) '%'])
acc_test = ComputeAccuracy(X_test, y_test, W, b);
disp(['Test Accuracy:' num2str(acc_test*100) '%'])

