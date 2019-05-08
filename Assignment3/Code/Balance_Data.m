function [X_balanced,Y_balanced] = Balance_Data(X_training, Y_training, ys_training, K)
    % Count the minimum number of samples from the minoritary class
%     [C,ia,ic] = unique(ys_training);
%     a_counts = accumarray(ic,1);
%     min_value = 20000;
%     for i=1:K
%         if a_counts(i)< min_value
%             min_value= a_counts(i);
%         end
%     end
    N_tr = size(X_training, 2); % Number of training samples
    min_value = N_tr; 
    for cl = 1:K
       positions{cl} = find(ys_training == cl);
       n_tmp = length(positions{cl});
       if n_tmp < min_value
           min_value = n_tmp;
       end
    end
    % Take min_value random samples from each class
    N_tr = min_value*K;
    X_balanced = zeros(size(X_training,1),N_tr);
    Y_balanced = zeros(size(Y_training,1),N_tr);
    for i=1:K
        positions_b = datasample(positions{i}, min_value);
        X_balanced(:,(i-1)*min_value+1: i*min_value) = X_training(:,positions_b) ;
        Y_balanced(:,(i-1)*min_value+1: i*min_value) = Y_training(:,positions_b) ;
    end
   
end