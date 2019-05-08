function MX = MakeMXMatrix(x_input,d,k,nf,optimize) 
 % x_input: input vector of sizex1
 %d= heigh of filter
 % k= weight filter
 % nf = number of filters
    nlen = length(x_input)/d;
    x_cols = d*k;
    x_rows = nlen-k+1;
    nf_cols = x_cols * nf;
    nf_rows = x_rows * nf;

    x_vecs = zeros(x_rows, x_cols);%, batch_size);
    MX = zeros(nf_rows, nf_cols);
    for i=1:x_rows
        pos_row = (i-1)*d + 1;
        x_vecs(i,:)= x_input(pos_row:pos_row + x_cols - 1);
    end
    
    if optimize == "True"
        MX = x_vecs;
    else
        for i=1:nf_rows
            pos_row = idivide(int32(i-1), int32(nf)) + 1;
            pos_col = mod((i-1),nf)* x_cols +1 ;
            MX(i,pos_col:pos_col+x_cols -1) = x_vecs(pos_row,:);
        end
    end

end