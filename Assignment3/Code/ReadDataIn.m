function ReadDataIn()
    load('assignment3_names.mat') %load all_names and labels (ys)
    C = unique(cell2mat(all_names));
    d = numel(C); %lenght d corresponds to dimensionality of one-hot vector
    max_nlen = 0;
    N = length(all_names);
    for i=1:N
        item = cell2mat(all_names(i));
        if length(item) > max_nlen
            max_nlen = length(item);
        end   
    end
    n_len = max_nlen;
    K = numel(unique(ys)); %number of classes to predict
    % intialize map : key is each character and value is the indicator/indice
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    for i=1:d
        char_to_ind(C(i)) = i;
    end

    % encode input names
    name_matrix = zeros(d, n_len);
    % each name into its matrix
    X = zeros(d*n_len, N);
    for i=1:N
        name_matrix = zeros(d, n_len);
        item = cell2mat(all_names(i));
        for j=1:length(item)
            one_hot_vector = zeros(d,1);
            index = values(char_to_ind,{item(j)});
            one_hot_vector(cell2mat(index)) = 1;
            name_matrix(:,j)= one_hot_vector;
        end
        name_vector = name_matrix(:);
        X(:,i) = name_vector;
    end

    disp('Saving the data')
    tic
    save('assignment3_X.mat', 'X');
    toc
end