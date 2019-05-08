% Exercise 1
% Read data in
load('assignment3_names.mat'); %load all_names and labels (ys)
load('assignment3_X.mat') %load X
C = unique(cell2mat(all_names));
d = numel(C); %lenght d corresponds to dimensionality of one-hot vector
n_len = size(X,1)/d;
K = numel(unique(ys)); %number of classes to predict
ys = ys';
% Split data into training and validation set
% Read Validation_Inds.txt
load('assignment3_indices.mat') %load ind_val
% Build validation dataset with the indices, and training is the remaining
% indices
N_val = length(ind_val);
N_total = length(all_names);
X_validation = zeros(d*n_len, N_val);%cell(1, N_val);
X_training = zeros(d*n_len,  N_total-N_val);%cell(1, N_total-N_val);
val=1;
train=1;
j=1;

% One-hot representation of Y
% ind = double(ys);
% vec = ind2vec(ind);
% Y1 =full(ys); % one-hot representation
%One-hot representation
Y = zeros(K,length(ys));
for i = 1:length(ys)
    vec = Y(:,i);
    pos = ys(i);
    vec(pos) = 1;
    Y(:,i) = vec;
end

ys_training = zeros(1, N_total-N_val);
ys_validation = zeros(1,N_val);
Y_validation = zeros(K, N_val);%cell(1, N_val);
Y_training = zeros(K, N_total-N_val);%cell(1, N_total-N_val);
for i=1:N_total
    item = X(:,i);
    item2 = Y(:,i);
    item3 =  ys(i);
    if i == ind_val(j)
        X_validation(:,val) = item;
        Y_validation(:,val) = item2;
        ys_validation(val)= item3;
        j=j+1;
        val = val + 1;
    else
        X_training(:,train) = item ;
        Y_training(:,train) = item2;
        ys_training(train)= item3;
        train = train +1;
    end
end

disp('Saving the data')
tic
save('assigment3_dataset.mat', 'X_training', 'X_validation', 'Y_training', 'Y_validation', ...
    'ys_training', 'ys_validation', 'N_total', 'K', 'd', 'n_len', 'C');
toc


%% Construct the convolution matrices check

load('DebugInfo.mat') ;
[d,k,nf]=size(F);
nlen = length(x_input)/d;
MF = MakeMFMatrix(F,nlen);
k1=k;
n1=nf;
ConvNet.F{1}=randn(d,k1,n1)*sqrt(2/(d+n1));
MX = MakeMXMatrix (x_input,d,k,nf,"False");
s1 = MX*F(:);
F2 = F(:);
s2 = MF*x_input;
S2= reshape(s2, [nf, nlen-4]);
S1= reshape(s1, [nf, nlen-4]);

 %% Gradients check
 
 % Set hyper-parameters & init convNet parameters
%load('DebugInfo.mat') ;
n1= 20;%number of filters at layer 1
k1 = 5; %Width of filters layer 1
n2= 20 ; %number of filters at layer 2
k2= 5;%Width of filters layer 2
ConvNet = InitConvNet(n1, n2, k1, k2, d, n_len, K);
batch_size = 100;
%X_batch = X_input(:,1:batch_size);
X_batch = X_training(:,1:batch_size);
Y_batch = Y_training(:,1:batch_size);

% Analytical gradients
[Pbatch,ConvNet]= Forward(ConvNet, X_batch);
[grad_W, grad_F1, grad_F2]  = Backward(ConvNet,Y_batch, Pbatch);

% numerical gradients
Gs = NumericalGradient(X_batch, Y_batch, ConvNet,1e-5);
ngrad_F1 = Gs{1};
ngrad_F2 = Gs{2};
ngrad_W = Gs{3};

% %reshape gradients F
% grad_F1= grad_F1(:);
% grad_F2= grad_F2(:);
% ngrad_F1 = ngrad_F1(:);
% ngrad_F2 = ngrad_F2(:);
eps =1e-4;
%check gradients
gradcheck_W = sum(abs(ngrad_W - grad_W)/max(eps, sum(abs(ngrad_W) + abs(grad_W))));
gradcheck_F1 = sum(sum(abs(ngrad_F1 - grad_F1)./max(eps, sum(sum(abs(ngrad_F1) + abs(grad_F1))))));
gradcheck_F2 = sum(sum(abs(ngrad_F2 - grad_F2)./max(eps, sum(sum(abs(ngrad_F2) + abs(grad_F2))))));

if gradcheck_W <= 1e-6 
    fprintf("Correct grad_W");
else
    fprintf("Incorrect grad_W");
end
if gradcheck_F1 <= 1e-3 
    fprintf("Correct grad_F1");
else
    fprintf("Incorrect grad_F1");
end
if gradcheck_F2 <= 1e-3 
    fprintf("Correct grad_F2");
else
    fprintf("Incorrect grad_F2");
end

%% Check of mini-batch without momentum
profile on

load('assigment3_dataset.mat');
% Set hyper-parameters & init convNet parameters
%load('DebugInfo.mat') ;
n1= 10;%number of filters at layer 1
k1 = 5; %Width of filters layer 1
n2= 10 ; %number of filters at layer 2
k2= 5;%Width of filters layer 2
ConvNet = InitConvNet(n1, n2, k1, k2, d, n_len, K);

n_update = 50;
n_batch = 100;
eta = 0.005;
n_epochs = 6;
N = size(X_training, 2); % Number of training samples
indices = 1:N;
nsteps = floor(N/n_batch);  % Number of steps per epoch
times = floor(nsteps*n_epochs/n_update); % Number of updates

J_training = zeros(1, times);
J_validation = zeros(1, times);
acc_train = zeros(1, times);
acc_val = zeros(1, times);
count=1;
updates = 1;

for i=1: n_epochs 
    % We randomize the batches selection 
    indices = indices(randperm(length(indices)));
    for j=1:nsteps
        % We take each random batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = indices(j_start:j_end);
        X_batch = X_training(:, inds);
        Y_batch = Y_training(:, inds);  
        % compute gradients for each mini-batch
        [Pbatch,ConvNet]= Forward(ConvNet, X_batch);
        [grad_W, grad_F1, grad_F2]  = Backward(ConvNet,Y_batch, Pbatch);
        % update weights and Filters with momentum
        ConvNet.W = ConvNet.W - eta*grad_W;
        ConvNet.F{1}  = ConvNet.F{1}  - eta*grad_F1;
        ConvNet.F{2}  = ConvNet.F{2} - eta*grad_F2; 

        % each n_updates we save loss and calculta accuracy
        if mod(updates,n_update) == 0
            J_training(count) = Compute_loss(X_training, Y_training, ConvNet);
            J_validation(count) = Compute_loss(X_validation, Y_validation, ConvNet);
            %Calculate accuracy
            acc_train(count) = ComputeAccuracy(X_training, ys_training, ConvNet);
            acc_val(count) = ComputeAccuracy(X_validation, ys_validation, ConvNet);
            fprintf("Update # %d\n", updates)
            fprintf("Training accuracy: %f\n", acc_train(count)*100)
            fprintf("Validation accuracy: %f\n", acc_val(count)*100)
            count = count+1;
        end
        updates = updates +1; 
    end
end

profile viewer
% Final Confusion Matrix on validation set
plotconfusion(Y_validation,Pred);

updates_vector1 = (1:size(J_training,2))*n_update;
updates_vector2 = (1:size(J_validation,2))*n_update;
%plot cost score
f1 = figure;
plot(updates_vector1 , J_training, 'g')
hold on
plot(updates_vector2, J_validation, 'r')
hold off
xlabel('Updates');
ylabel('Loss');
legend('Training loss', 'Validation loss');
saveas(f1,'Figures/cost_unbalanced.png','png')

%plot accuracy

updates_vector1 = (1:size(acc_train,2))*n_update;
updates_vector2 = (1:size(acc_val,2))*n_update;
f2 = figure;
plot(updates_vector1, acc_train, 'g')
hold on
plot(updates_vector2, acc_val, 'r')
hold off
xlabel('Updates');
ylabel('Accuracy');
legend('Training accuracy', 'Validation accuracy');
saveas(f2,'Figures/acc_unbalanced.png','png')

%% Training 1: Unbalanced data
profile on
load('assigment3_dataset.mat');
% Set hyper-parameters & init convNet parameters
%load('DebugInfo.mat') ;
n1= 20;%number of filters at layer 1
k1 = 5; %Width of filters layer 1
n2= 20 ; %number of filters at layer 2
k2= 3;%Width of filters layer 2
rho = 0.9; %momentum
ConvNet = InitConvNet(n1, n2, k1, k2, d, n_len, K);

n_update = 50;
n_batch = 100;
eta = 0.005;
n_epochs = 112;% per fer 20000 updates, 40 saved
N = size(X_training, 2);
indices = 1:N;
nsteps = floor(N/n_batch);  % Number of steps per epoch
times = floor(nsteps*n_epochs/n_update); % Number of updates
J_training = zeros(1, times);
J_validation = zeros(1, times);
acc_train = zeros(1, times);
acc_val = zeros(1, times);
count=1;
updates = 1;
for i=1: n_epochs 
    v_W = zeros(size(ConvNet.W));
    v_F = {zeros(size(ConvNet.F{1})), zeros(size(ConvNet.F{2}))};
    % We randomize the batches selection 
    indices = indices(randperm(length(indices)));
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = indices(j_start:j_end);
        X_batch = X_training(:, inds);
        Y_batch = Y_training(:, inds);  
        % compute gradients for each mini-batch
        [Pbatch,ConvNet]= Forward(ConvNet, X_batch);
        [grad_W, grad_F1, grad_F2]  = Backward(ConvNet,Y_batch, Pbatch);
        
        % update weights and Filters with momentum
        v_W = rho*v_W + eta*grad_W;
        v_F{1} = rho*v_F{1} + eta*grad_F1;
        v_F{2} = rho*v_F{2} + eta*grad_F2;
        ConvNet.W = ConvNet.W - v_W;
        ConvNet.F{1}  = ConvNet.F{1} - v_F{1} ;
        ConvNet.F{2}  = ConvNet.F{2}  -  v_F{2} ;   

        % each n_updates we save loss and calculta accuracy
        if mod(updates,n_update) == 0
            J_training(count) = Compute_loss(X_training, Y_training, ConvNet);
            J_validation(count) = Compute_loss(X_validation, Y_validation, ConvNet);
            %Calculate accuracy
            acc_train(count) = ComputeAccuracy(X_training, ys_training, ConvNet);
            acc_val(count) = ComputeAccuracy(X_validation, ys_validation, ConvNet);
            fprintf("Update # %d\n", updates)
            fprintf("Training accuracy: %f\n", acc_train(count)*100)
            fprintf("Validation accuracy: %f\n", acc_val(count)*100)
            count = count+1;
        end
        updates = updates +1; 
    end
end

profile viewer
% Final Confusion Matrix on validation set
P = Evaluate(X_validation, ConvNet.W, ConvNet.MF{1},ConvNet.MF{2});
[argvalue, argmax]  = max(P);% argmax contains the class predictions
predictions = argmax;
Conf = confusionmat(ys_validation,predictions);
plotConfMat(Conf);

updates_vector1 = (1:size(J_training,2))*n_update;
updates_vector2 = (1:size(J_validation,2))*n_update;
%plot cost score
f1 = figure;
plot(updates_vector1 , J_training, 'g')
hold on
plot(updates_vector2, J_validation, 'r')
hold off
xlabel('Updates');
ylabel('Loss');
legend('Training loss', 'Validation loss');
saveas(f1,'Figures/cost_unbalanced_momentum.png','png')

%plot accuracy
updates_vector1 = (1:size(acc_train,2))*n_update;
updates_vector2 = (1:size(acc_val,2))*n_update;
f2 = figure;
plot(updates_vector1, acc_train, 'g')
hold on
plot(updates_vector2, acc_val, 'r')
hold off
xlabel('Updates');
ylabel('Accuracy');
legend('Training accuracy', 'Validation accuracy');
saveas(f2,'Figures/acc_unbalanced_momentum.png','png')

disp('Saving the data')
tic
Acc_train_f = acc_train(size(acc_train,2));
Acc_val_f = acc_val(size(acc_val,2));
J_train_f = J_training(size(J_training,2));
J_val_f= J_validation(size(J_validation,2));
save('assignment3_unbalanced_20000updates.mat', 'Acc_train_f','Acc_val_f', 'J_train_f', 'J_val_f','Conf');
toc

%% Training 2
profile on
load('assigment3_dataset.mat');
% Set hyper-parameters & init convNet parameters
%load('DebugInfo.mat') ;
n1= 20;%number of filters at layer 1
k1 = 5; %Width of filters layer 1
n2= 20 ; %number of filters at layer 2
k2= 3;%Width of filters layer 2
rho = 0.9; %momentum
ConvNet = InitConvNet(n1, n2, k1, k2, d, n_len, K);

n_update = 500;
n_batch = 100;
eta = 0.005;
n_epochs = 1667;%150;
[X_,Y_] = Balance_Data(X_training, Y_training,ys_training, K);
N = size(X_, 2);
indices = 1:N;
nsteps = floor(N/n_batch);  % Number of steps per epoch
times = floor(nsteps*n_epochs/n_update); % Number of updates
J_training = zeros(1, times);
J_validation = zeros(1, times);
acc_train = zeros(1, times);
acc_val = zeros(1, times);
count=1;
updates = 1;

for i=1: n_epochs 
    v_W = zeros(size(ConvNet.W));
    v_F = {zeros(size(ConvNet.F{1})), zeros(size(ConvNet.F{2}))};
    %Balance data
    [X_balanced,Y_balanced] = Balance_Data(X_training,Y_training,ys_training, K);
    % We randomize the batches selection 
    indices = indices(randperm(length(indices)));
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = indices(j_start:j_end);
        X_batch = X_balanced(:, inds);
        Y_batch = Y_balanced(:, inds);  
        % compute gradients for each mini-batch
        [Pbatch,ConvNet]= Forward(ConvNet, X_batch);
        [grad_W, grad_F1, grad_F2]  = Backward(ConvNet,Y_batch, Pbatch);
         
        % update weights and Filters with momentum
        v_W = rho*v_W + eta*grad_W;
        v_F{1} = rho*v_F{1} + eta*grad_F1;
        v_F{2} = rho*v_F{2} + eta*grad_F2;
        ConvNet.W = ConvNet.W - v_W;
        ConvNet.F{1}  = ConvNet.F{1} - v_F{1} ;
        ConvNet.F{2}  = ConvNet.F{2}  -  v_F{2} ;  
        
        % each n_updates we save loss and calculta accuracy
        if mod(updates,n_update) == 0
            J_training(count) = Compute_loss(X_training, Y_training, ConvNet);
            J_validation(count) = Compute_loss(X_validation, Y_validation, ConvNet);
            %Calculate accuracy
            acc_train(count) = ComputeAccuracy(X_training, ys_training, ConvNet);
            acc_val(count) = ComputeAccuracy(X_validation, ys_validation, ConvNet);
            fprintf("Update # %d\n", updates)
            fprintf("Training accuracy: %f\n", acc_train(count)*100)
            fprintf("Validation accuracy: %f\n", acc_val(count)*100)
            count = count+1;
        end
        updates = updates +1; 
    end
end

profile viewer

updates_vector1 = (1:size(J_training,2))*n_update;
updates_vector2 = (1:size(J_validation,2))*n_update;
%plot cost score
f1 = figure;
plot(updates_vector1 , J_training, 'g')
hold on
plot(updates_vector2, J_validation, 'r')
hold off
xlabel('Updates');
ylabel('Loss');
legend('Training loss', 'Validation loss');
saveas(f1,'Figures/cost_balanced_momentum.png','png')

%plot accuracy
updates_vector1 = (1:size(acc_train,2))*n_update;
updates_vector2 = (1:size(acc_val,2))*n_update;
f2 = figure;
plot(updates_vector1, acc_train, 'g')
hold on
plot(updates_vector2, acc_val, 'r')
hold off
xlabel('Updates');
ylabel('Accuracy');
legend('Training accuracy', 'Validation accuracy');
saveas(f2,'Figures/acc_balanced_momentum.png','png')

% Final Confusion Matrix on validation set
P = Evaluate(X_validation, ConvNet.W, ConvNet.MF{1},ConvNet.MF{2});
[argvalue, argmax]  = max(P);% argmax contains the class predictions
predictions = argmax;
Conf = confusionmat(ys_validation,predictions);
plotConfMat(Conf);

disp('Saving the data')
tic
Acc_train_f = acc_train(size(acc_train,2));
Acc_val_f = acc_val(size(acc_val,2));
J_train_f = J_training(size(J_training,2));
J_val_f= J_validation(size(J_validation,2));
save('assignment3_balanced_20000updates.mat', 'Acc_train_f','Acc_val_f', 'J_train_f', 'J_val_f','Conf');
toc

%% Training 3: best network. balanced data
profile on
load('assigment3_dataset.mat');
% Set hyper-parameters & init convNet parameters
%load('DebugInfo.mat') ;
n1= 20;%number of filters at layer 1
k1 = 5; %Width of filters layer 1
n2= 40 ; %number of filters at layer 2
k2= 3;%Width of filters layer 2
rho = 0.9; %momentum
ConvNet = InitConvNet(n1, n2, k1, k2, d, n_len, K);

n_update = 500;
n_batch = 100;
eta = 0.005;
n_epochs = 500;
[X_,Y_] = Balance_Data(X_training, Y_training,ys_training, K);
N = size(X_, 2);
indices = 1:N;
nsteps = floor(N/n_batch);  % Number of steps per epoch
times = floor(nsteps*n_epochs/n_update); % Number of updates


J_training = zeros(1, times);
J_validation = zeros(1, times);
acc_train = zeros(1, times);
acc_val = zeros(1, times);
count=1;
updates = 1;

for i=1: n_epochs 
    v_W = zeros(size(ConvNet.W));
    v_F = {zeros(size(ConvNet.F{1})), zeros(size(ConvNet.F{2}))};
    %Balance data
    [X_balanced,Y_balanced] = Balance_Data(X_training,Y_training,ys_training, K);
    % We randomize the batches selection 
    indices = indices(randperm(length(indices)));
    for j=1:N/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = indices(j_start:j_end);
        X_batch = X_balanced(:, inds);
        Y_batch = Y_balanced(:, inds);  
        % compute gradients for each mini-batch
        [Pbatch,ConvNet]= Forward(ConvNet, X_batch);
        [grad_W, grad_F1, grad_F2]  = Backward(ConvNet,Y_batch, Pbatch);
         
        % update weights and Filters with momentum
        v_W = rho*v_W + eta*grad_W;
        v_F{1} = rho*v_F{1} + eta*grad_F1;
        v_F{2} = rho*v_F{2} + eta*grad_F2;
        ConvNet.W = ConvNet.W - v_W;
        ConvNet.F{1}  = ConvNet.F{1} - v_F{1} ;
        ConvNet.F{2}  = ConvNet.F{2}  -  v_F{2} ;  
        
        % each n_updates we save loss and calculta accuracy
        if mod(updates,n_update) == 0
            J_training(count) = Compute_loss(X_training, Y_training, ConvNet);
            J_validation(count) = Compute_loss(X_validation, Y_validation, ConvNet);
            %Calculate accuracy
            acc_train(count) = ComputeAccuracy(X_training, ys_training, ConvNet);
            acc_val(count) = ComputeAccuracy(X_validation, ys_validation, ConvNet);
            fprintf("Update # %d\n", updates)
            fprintf("Training accuracy: %f\n", acc_train(count)*100)
            fprintf("Validation accuracy: %f\n", acc_val(count)*100)
            count = count+1;
        end
        updates = updates +1; 
    end
end
profile viewer
updates_vector1 = (1:size(J_training,2))*n_update;
updates_vector2 = (1:size(J_validation,2))*n_update;
%plot cost score
f1 = figure;
plot(updates_vector1 , J_training, 'g')
hold on
plot(updates_vector2, J_validation, 'r')
hold off
xlabel('Updates');
ylabel('Loss');
legend('Training loss', 'Validation loss');
saveas(f1,'Figures/cost_balanced_momentum.png','png')

%plot accuracy
updates_vector1 = (1:size(acc_train,2))*n_update;
updates_vector2 = (1:size(acc_val,2))*n_update;
f2 = figure;
plot(updates_vector1, acc_train, 'g')
hold on
plot(updates_vector2, acc_val, 'r')
hold off
xlabel('Updates');
ylabel('Accuracy');
legend('Training accuracy', 'Validation accuracy');
saveas(f2,'Figures/acc_balanced_momentum.png','png')

% Final Confusion Matrix on validation set
P = Evaluate(X_validation, ConvNet.W, ConvNet.MF{1},ConvNet.MF{2});
[argvalue, argmax]  = max(P);% argmax contains the class predictions
predictions = argmax;
Conf = confusionmat(ys_validation,predictions);
plotConfMat(Conf);

disp('Saving the data')
tic
Acc_train_f = acc_train(size(acc_train,2));
Acc_val_f = acc_val(size(acc_val,2));
J_train_f = J_training(size(J_training,2));
J_val_f= J_validation(size(J_validation,2));
save('assignment3_balanced_menysupdates.mat', 'Acc_train_f','Acc_val_f', 'J_train_f', 'J_val_f','Conf', 'P','ConvNet');
toc


%% Performance of the network for 5 friends surnames (international)

load('assigment3_dataset.mat');
surnames = {'Gonzalez','Sieger','Tomasini','Moura','Perier','Canal'};
y_surnames = [17,7,10,14,6,17];
categories = {'Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', ...
    'German', 'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', ...
    'Portuguese', 'Russian', 'Scottish', 'Spanish', 'Vietnamese'};

% intialize map : key is each character and value is the indicator/indice
char_to_ind = containers.Map('KeyType','char','ValueType','int32');
for i=1:d
    char_to_ind(C(i)) = i;
end
% encode input names
name_matrix = zeros(d, n_len);
% each name into its matrix
N_friends= length(surnames);
X_surnames = zeros(d*n_len, N_friends);
for i=1:N_friends
    name_matrix = zeros(d, n_len);
    item = cell2mat(surnames(i));
    for j=1:length(item)
        one_hot_vector = zeros(d,1);
        index = values(char_to_ind,{item(j)});
        one_hot_vector(cell2mat(index)) = 1;
        name_matrix(:,j)= one_hot_vector;
    end
    name_vector = name_matrix(:);
    X_surnames(:,i) = name_vector;
end

% Get my friends probabilities
[P_friends,ConvNet] = Forward(ConvNet, X_surnames);

f = image(P_friends, 'CDataMapping','scaled');
colorbar()
xticks(1:6)
xticklabels(surnames)
yticks(1:18)
yticklabels(categories)
title('Surnames probabilities')
colormap copper
saveas(f,'Figures/friends_probabilities','png')
% Calculate accuracy
acc = ComputeAccuracy(X_surnames, y_surnames, ConvNet);
fprintf("Accuracy: %f\n", acc*100)


