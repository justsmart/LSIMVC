function [Con_P,obj] = LSIMVC4(X,W_graph,G,ind_folds,numClust,lambda,beta,para_r,max_iter)
% sparse

numInst = size(ind_folds,1);
% ---------- ³õÊ¼»¯ ---------- %
rand('seed',7578);
alpha = ones(length(X),1);
alpha_r = alpha.^para_r;

con_P = 0;
Piv = cell(1,length(X));
for iv = 1:length(X)
    linshi_U = rand(size(X{iv},1),numClust);
    U{iv} = orth(linshi_U);
    Piv{iv} = U{iv}'*X{iv};
    con_P = con_P + Piv{iv}*G{iv}';
end
% Con_P = con_P./repmat(sum(ind_folds,2)',numClust,1);
Con_P = rand(numClust,numInst);
clear con_P

for iter = 1:max_iter
    
    % ---------------- Con_P ---------- %    
    linshi_GDG = 0;
    linshi_PWG = 0;
    for iv = 1:length(X)
        graph_A = diag(sum(W_graph{iv},2));
        linshi_GDG = linshi_GDG+alpha_r(iv)*G{iv}*graph_A*G{iv}';
        linshi_PWG = linshi_PWG+alpha_r(iv)*Piv{iv}*W_graph{iv}*G{iv}';
    end
    Con_P = linshi_PWG*diag(1./max(diag(linshi_GDG),1e-8));   
%     for in = 1:size(Con_P,2)
%         temp1 = linshi_Con_P(:,in);
%         temp2 = 0.5*beta/lambda/linshi_GDG(in,in);
%         Con_P(:,in) = max(0,temp1-temp2)+min(0,temp1+temp2);
%     end    
        
    
    % -------- U P ------ %
    for iv = 1:length(X)
        temp = X{iv}*Piv{iv}';
        temp(isnan(temp)) = 0;
        temp(isinf(temp)) = 0;        
        [linshi_U,~,linshi_V] = svd(temp,'econ');        
        linshi_U(isnan(linshi_U)) = 0;
        linshi_U(isinf(linshi_U)) = 0;    
        linshi_V(isnan(linshi_V)) = 0;
        linshi_V(isinf(linshi_V)) = 0; 
        U{iv} = linshi_U*linshi_V';  
    end


    % -------- P -----%
    for iv = 1:length(X)             
        graph_D = lambda*sum(W_graph{iv},1)+1;
        linshi_P = (lambda*Con_P*G{iv}*W_graph{iv}+U{iv}'*X{iv})*diag(1./graph_D); 
        
            
%         for in = 1:size(linshi_P,2)
%             linshi_P(:,in) = EProjSimplex_new(linshi_P(:,in));
%         end    
        for in = 1:size(linshi_P,2)
            temp1 = linshi_P(:,in);
            temp2 = 0.5*beta/graph_D(in);
            linshi_P(:,in) = max(0,temp1-temp2)+min(0,temp1+temp2);
        end  
        Piv{iv} = linshi_P;
    end
    % ----------- alpha--------%
    for iv = 1:length(X)
        graph_D = diag(sum(W_graph{iv}));
        val = trace(Piv{iv}*graph_D*Piv{iv}')+trace(Con_P*G{iv}*graph_D*G{iv}'*Con_P')-2*trace(Con_P*G{iv}*W_graph{iv}*Piv{iv}');
        Rec_error(iv) = norm(X{iv}-U{iv}*Piv{iv},'fro')^2+lambda*val+beta*sum(sum(abs(Piv{iv})));
    end
    aH = bsxfun(@power,Rec_error, 1/(1-para_r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,aH,sum(aH)); % alpha = H./sum(H);
    alpha_r = alpha.^para_r;   
    % ------- obj ------- %
    obj(iter) = alpha_r*Rec_error';
    if iter > 3 && abs(obj(iter)-obj(iter-1))<1e-6
        iter
        break;
    end
end