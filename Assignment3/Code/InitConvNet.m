function ConvNet = InitConvNet(n1, n2, k1, k2, d, n_len, K)
    ConvNet.d = d;
    ConvNet.K = K;
    % He initialization
    sig1= sqrt(2/(d+n1)); %n or k
    sig2= sqrt(2/n1);
    sigW= sqrt(2/n2);
    nlen1 = n_len-k1+1; 
    nlen2 = nlen1-k2+1; 
    ConvNet.F{1}=randn(d,k1,n1)*sig1;
    ConvNet.F{2}=randn(n1,k2,n2)*sig2;
    fsize = n2*nlen2; %number of elements of X(2)
    ConvNet.W= randn(K,fsize)*sigW;
end