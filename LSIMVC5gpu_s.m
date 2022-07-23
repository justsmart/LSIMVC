function [Con_P,obj] = LSIMVC5gpu_s(X,W_graph,G,ind_folds,numClust,lambda,beta,para_r,max_iter)
% sparse
X = togpu_s(X);
W_graph = togpu_s(W_graph);
G = togpu_s(G);
ind_folds = togpu_s(ind_folds);
% numClust = togpu(numClust);
% lambda = togpu(lambda);
% beta = togpu(beta);
% para_r = togpu(para_r);
rng(1)


numInst = size(ind_folds,1);
% ---------- ³õÊ¼»¯ ---------- %
% rng('default')
% rng(7578);

% rand('seed',7578);
alpha = ones(length(X),1,'single','gpuArray');
alpha_r = alpha.^para_r;

% con_P = 0;
Piv = cell(1,length(X));
U = cell(1,length(X));
for iv = 1:length(X)
    linshi_U = rand(size(X{iv},1),numClust,'single','gpuArray');
    U{iv} = orth(linshi_U);
    Piv{iv} = U{iv}'*X{iv};
%     con_P = con_P + Piv{iv}*G{iv}';
end
% Con_P = con_P./repmat(sum(ind_folds,2)',numClust,1);
% Con_P = rand(numClust,numInst,'gpuArray');
% clear con_P
obj = zeros(1,max_iter,'single','gpuArray');
for iter = 1:max_iter
%     iter
%     tic;
    % ---------------- Con_P ---------- %
    linshi_GDG = 0;
    linshi_PWG = 0;
    for iv = 1:length(X)
        graph_A = sum(W_graph{iv},2);
        aa = alpha_r(iv)*graph_A;
        
        cc = zeros(size(ind_folds,1),1,'single','gpuArray');
        cc(ind_folds(:,iv)==1) = aa;
        linshi_GDG = linshi_GDG+cc;
        linshi_PWG = linshi_PWG+alpha_r(iv)*Piv{iv}*W_graph{iv}*G{iv}';
    end
    Con_P = linshi_PWG*diag(1./max(linshi_GDG,1e-8));
    %--Con_P*G--%
    Con_PxG = cell(length(X),1);
    for iv = 1:length(X)
        Con_PxG{iv} = Con_P*G{iv};
    end
    
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
    %     W_graph= togpu(W_graph);
    %     Con_P= togpu(Con_P);
    %     U= togpu(U);
    %     X= togpu(X);
    %     G= togpu(G);
    
    % -------- P -----%
    
    for iv = 1:length(X)
        graph_D = lambda*sum(W_graph{iv},1)+1;
        linshi_P = (lambda*Con_PxG{iv}*W_graph{iv}+U{iv}'*X{iv})*diag(1./graph_D);
        
        %         for in = 1:size(linshi_P,2)
        %             linshi_P(:,in) = EProjSimplex_new(linshi_P(:,in));
        %         end
        
        temp = 0.5*beta./graph_D;
        
        linshi_P = max(0,linshi_P-temp)+min(0,linshi_P+temp);
        %         for in = 1:size(linshi_P,2)
        %             temp1 = linshi_P(:,in);
        %             temp2 = 0.5*beta/graph_D(in);
        %
        %             linshi_P(:,in) = max(0,temp1-temp2)+min(0,temp1+temp2);
        %         end
        Piv{iv} = linshi_P;
    end
    
    
    % ----------- alpha--------%
    Rec_error = zeros(1,length(X),'single','gpuArray');
%     Rec_error = togpu(Rec_error);
    
    for iv = 1:length(X)
        %         Piv{iv}= togpu(Piv{iv});
        %         W_graph{iv}= togpu(W_graph{iv});
        %         Con_P = togpu(Con_P);
        %         G{iv} = togpu(G{iv});
        %         tic;
%         graph_D = diag(sum(W_graph{iv}));
%         t1=trace(Piv{iv}*graph_D*Piv{iv}');
%         t2=trace(Con_P*G{iv}*graph_D*G{iv}'*Con_P');
%         t3=trace(Con_P*G{iv}*W_graph{iv}*Piv{iv}');
%         
%         
%         val = t1+t2-2*t3;
        val = getVal(W_graph{iv},Piv{iv},Con_PxG{iv});
        
        %         Piv{iv}= gather(Piv{iv});
        %         W_graph{iv}= gather(W_graph{iv});
        %         Con_P = gather(Con_P);
        %         G{iv} = gather(G{iv});
        %         class(X{iv})
        %         class(U{iv})
        %         class(Piv{iv})
        Rec_error(iv) = abs(norm(X{iv}-U{iv}*Piv{iv},'fro')^2+lambda*val+beta*sum(sum(abs(Piv{iv}))));
        %         toc;
    end
    
    aH = bsxfun(@power,Rec_error, 1/(1-para_r));     % h = h.^(1/(1-r));
    alpha = bsxfun(@rdivide,aH,sum(aH)); % alpha = H./sum(H);
    alpha_r = alpha.^para_r;
    % ------- obj ------- %
    obj(iter) = alpha_r*Rec_error';
%     toc;
    if iter > 3 && abs(obj(iter)-obj(iter-1))<1e-6
        
        obj = obj(1:iter);
        
        break;
    end
    
end
Con_P=gather(Con_P);
obj=gather(obj);
end


function val = getVal(w,p,cg)
%w is W_graph{iv} ,p is Piv{iv},cg is Con_PxG
graph_D = diag(sum(w));
pd=p*graph_D;
t1=trace(pd*p');

t2=trace(cg*graph_D*cg');
t3=trace(cg*w*p');
val = t1+t2-2*t3;
end