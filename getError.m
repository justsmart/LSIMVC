function val = getError(w,p,Con_P,g)

graph_D = diag(sum(w));
pd=p*graph_D;
t1=trace(pd*p');
cg=Con_P*g;
t2=trace(cg*graph_D*cg');
t3=trace(cg*w*p');
% val=1;
val = t1+t2-2*t3;

end