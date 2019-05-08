function [grad_W, grad_F1, grad_F2]  = Backward(ConvNet,Y_batch, Pbatch)
    % BACKWARD
    grad_W = zeros(size(ConvNet.W));
    grad_F1= zeros(size(ConvNet.F{1}));
    grad_F2 = zeros(size(ConvNet.F{2}));

    Gbatch = Pbatch-Y_batch;
    n = size(Pbatch,2);
    grad_W = Gbatch*ConvNet.X_batch{2}'/n;
   
    %Propagate gradient
    Gbatch= ConvNet.W'*Gbatch;
    Gbatch=Gbatch.*logical(ConvNet.X_batch{2});

    % gradients layer 2
    [n1,k2,n2]=size(ConvNet.F{2});
    %grad_F2= zeros(1,n1*k2*n2);
    for j = 1 : size(Gbatch, 2)
        g_j = Gbatch(:,j);
        x_j = ConvNet.X_batch{1}(:,j);
        % Calculate MX
        %[n1,k2,n2]=size(ConvNet.F{2});
        MX2 = MakeMXMatrix (x_j,n1,k2,n2,"False");
        v =  g_j'*MX2/n;  
        %n = size(ConvNet.X_batch{1}, 2);
        v1 = reshape(v,[n1, k2, n2]);
        grad_F2 = grad_F2+ v1;
    end
    %grad_F2 = reshape(grad_F2,[n1, k2, n2]);
    %Propagate gradient
    Gbatch= ConvNet.MF{2}'*Gbatch;
    Gbatch=Gbatch.*logical(ConvNet.X_batch{1});

     % gradients layer 1
    [d,k1,n1]=size(ConvNet.F{1});
    %grad_F1= zeros(1,d*k1*n1);
    for j = 1 : size(Gbatch, 2)
        g_j = Gbatch(:,j);
        x_j = ConvNet.Xbatch(:,j);
        % Calculate MX
        %[d,k1,n1]=size(ConvNet.F{1});
        MX1 = MakeMXMatrix (x_j,d,k1,n1,"False");
        v =  g_j'*MX1/n;  
        %n = size(ConvNet.Xbatch, 2);
        v1 = reshape(v,[d, k1, n1]);
        grad_F1 = grad_F1+ v1;
    end
    %grad_F1 = reshape(grad_F1, [d, k1, n1]);
end