fid = fopen('Validation_Inds.txt','r');
S = fscanf(fid,'%d');
fclose(fid);
ind_val= S;

disp('Saving the data')
tic
save('assignment3_indices.mat', 'ind_val');
toc