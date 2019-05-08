function MF = MakeMFMatrix(F, nlen)
% Function outputs the matrix Mf of size (nlen-k+1)*nf x nlen*dd
    [dd,k,nf]=size(F);
    r_size = (nlen-k+1)*nf;
    MF = zeros(r_size , nlen*dd);
    vf = zeros(nf,dd*k);
    % Build Vf= vec(Fn)'
    for i=1:nf
        item = F(:,:,i);
        vf(i,:) = item(:)';
    end
    for i=1:r_size
        pos_col = idivide(int32(i-1), int32(nf))*dd + 1;
        pos_row = mod((i-1),nf) +1;
        MF(i ,pos_col: (pos_col + k*dd -1))= vf(pos_row,:);
    end
end
