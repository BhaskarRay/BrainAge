group_ica_fmri=load('c_n15.mat');

for i=1:1113
    sym=group_ica_fmri.c_n(:,:,i);
    symU=triu(sym);
    v=nonzeros(symU);
    v=v';
    symL=tril(sym);
    v_l=nonzeros(symL);
    v_l=v_l';
    fnc_u(i,:)=v;
    fnc_l(i,:)=v_l;
end
save('fnc_u.mat','fnc_u');
save('fnc_l.mat','fnc_l');
dlmwrite('fnc_u.csv',fnc_u,'delimiter',',','precision',100);
dlmwrite('fnc_l.csv',fnc_l,'delimiter',',','precision',100);