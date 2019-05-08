%% Exercise1
% Read in data
book_fname = 'goblet_book.txt';
fid = fopen(book_fname, 'r');
book_data = fscanf(fid, '%c');
fclose(fid);
%vector containing the unique characters in book data:
book_chars = unique(book_data);
K = length(book_chars);   % length of unique characters
% Create map containers
value = 1:K;
key = num2cell(book_chars);
char_to_ind = containers.Map(key, value);
ind_to_char = containers.Map(value, key);

% Init hyper-parameters         
m = 100;                    % dimensionality of hidden states
eta = 0.1;                  % learning rate
seq_length = 25;            % length of input sequence
sig = 0.01;                 % sigma
% bias vectors  
RNN.b = zeros(m, 1);             
RNN.c = zeros(K, 1);
% weight matrices
RNN.U = randn(m, K)*sig;    
RNN.W = randn(m, m)*sig;
RNN.V = randn(K, m)*sig;
M.U = zeros(size(RNN.U));
M.W = zeros(size(RNN.W));
M.V = zeros(size(RNN.V));
M.b = zeros(size(RNN.b));
M.c = zeros(size(RNN.c));


%% Gradients check
x = zeros(1, length(book_data));
for i = 1:length(book_data)        
    x(i) = char_to_ind(book_data(i));
end
vec = ind2vec(x);
X =full(vec); % one-hot representation
batch_size = 25;
Xg = X(:, 1:seq_length);
Yg= X(:, 2:seq_length + 1);
% numerical gradients
num_grads = ComputeGradsNum(Xg(:, 1:batch_size), Yg(:, 1:batch_size), RNN, 1e-4);
% analytical gradients
h0 = zeros(size(RNN.W, 1), 1);
[~, a, h, ~, p] = forward(RNN, Xg(:,1:batch_size), Yg(:,1:batch_size),h0, batch_size, K, m);
grads = ComputeGradients(RNN, Xg(:,1:batch_size), Yg(:, 1:batch_size),a, h, p, batch_size, m);

% relative error rate
eps = 1e-5;
for f = fieldnames(RNN)'
    num_g = num_grads.(f{1});
    g = grads.(f{1});
    denominator = abs(num_g) + abs(g);
    numerator = abs(num_g - g);
    gradcheck = sum(numerator(:))/max(eps, sum(denominator(:)));
    
    disp(['Field name: ' f{1}]);
    if gradcheck <= 1e-6 
        disp(['Correct gradients: ' num2str(gradcheck)]);
    else
        disp(['Incorrect gradients: ' num2str(gradcheck)]);
    end
    
end

%% Training RNN using AdaGrad

x= zeros(1, length(book_data));
for i = 1:length(book_data)
    x(i) = char_to_ind(book_data(i));
end
vec = ind2vec(x);
data =full(vec); % one-hot representation   
iter = 1;
n_epochs = 10;
smoothLoss = [];
sloss = 0;
hprev = [];
min_loss = 200;
for i = 1 : n_epochs
    [RNN, sloss, iter, M, min_RNN, min_h, min_iter, min_loss] = MiniBatchGD(RNN, ...
        data, seq_length, K, m, eta, iter, M, ind_to_char, sloss(end), min_loss);
    smoothLoss = [smoothLoss, sloss];
end

%% Plot smooth loss function
f1 = figure;
plot((1:length(smoothLoss))*100, smoothLoss);
xlabel('Updates');
ylabel('Loss');
title('Smooth loss function');
saveas(f1,'Figures/smooth_loss_3.png','png')

%% Print best model
text_length = 1000;
y = synthesize(min_RNN, min_h, data(:, 1), text_length, K);
c = [];
for i = 1:text_length
    c = [c ind_to_char(y(i))];
end
fprintf('\n');
disp(['iter = ' num2str(min_iter) ', minimum loss = ' num2str(min_loss)]);
disp(c);



