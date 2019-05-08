function [Pbatch,ConvNet] = Forward(ConvNet, X_batch)
    % FORWARD
    %Construct MF first layer
    [d,k1,n1]=size(ConvNet.F{1});
%     nlen1 = size(X_batch,2);
%     ConvNet.Xbatch = X_batch(:); 
    nlen1 = size(X_batch,1)/d;
    ConvNet.Xbatch = X_batch;
    ConvNet.MF{1}= MakeMFMatrix(ConvNet.F{1},nlen1);
    % Construct MF second layer
    [n1,k2,n2]=size(ConvNet.F{2});
    nlen2 = nlen1-k1+1;
    ConvNet.MF{2} = MakeMFMatrix(ConvNet.F{2},nlen2);
    %MF = {MF1,MF2};
    % Apply 1st convolutional layer
    ConvNet.X_batch{1} = max(ConvNet.MF{1}*ConvNet.Xbatch,0);
    % Apply 2nd convolutional layer
    ConvNet.X_batch{2} = max(ConvNet.MF{2}*ConvNet.X_batch{1},0);
    % Apply fully connected matrix
    %Sbatch = ConvNet.W(:,1:size(ConvNet.X_batch{2},1))*ConvNet.X_batch{2};
    Sbatch = ConvNet.W*ConvNet.X_batch{2};
    denorm = repmat(sum(exp(Sbatch), 1), size(ConvNet.W, 1), 1);
    Pbatch = exp(Sbatch)./denorm;
       
end