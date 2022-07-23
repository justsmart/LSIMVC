clear
clc
rng('default')
rng(1);
addpath('../../../MATLAB-NOUPLOAD/cluster-data/data/');
addpath('../../../MATLAB-NOUPLOAD/cluster-data/incomplete/');
addpath('utils/');
Dataname = '3sources';
resPath = './RES/';

MODE = 'percentDel';
iter_num = 5;
for percentDel = [0.5]
    
    Datafold = [Dataname,'_',MODE,'_',num2str(percentDel),'.mat'];
    [X,truth,folds]=loadDataset(Dataname,Datafold);

    [numFold,numInst] = size(folds);
    
    numClust = length(unique(truth));
    numInst  = length(truth);
    
    
 
    truthF = truth;

    indicator = {'acc','nmi','pur','F_score','P_score','R_score','nmi_score','avgent_score','AR_score','RIi_score'};
    best_indicator = zeros(1,length(indicator));
    for_count=0;
    options = [];
    options.NeighborMode = 'KNN';
    
    options.WeightMode = 'HeatKernel';      % Binary  HeatKernel
    text_name = [resPath,Dataname,'_',num2str(percentDel),'_',options.WeightMode,'_','LSIMVC5_',num2str(iter_num),'.txt'];
    if exist(text_name,'file')==0
        b=fopen(text_name,'a+');
        fprintf(b,'iter mean_acc std_acc mean_nmi std_nmi mean_pur std_pur F_score P_score R_score nmi_score avgent_score AR_score RIi_score lambda beta para_r para_k');
        fprintf(b,'\n');
        fclose(b);
        para_exists=textscan(fopen(text_name),'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','headerlines',1);
        para_exists = para_exists(end-3:end);
    else
        
        para_exists=textscan(fopen(text_name),'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f','headerlines',1);
        para_exists = para_exists(end-3:end);

    end
    
    

    for lambda = [1e0]
        for beta = [1e-2]
            for para_r = [2]
                for para_k = [5]
                   
                    
                    for_count=for_count+1;
                    acc = zeros(1,iter_num);
                    nmi = zeros(1,iter_num);
                    pur = zeros(1,iter_num);
                    Fi = zeros(1,iter_num);
                    Pi = zeros(1,iter_num);
                    Ri = zeros(1,iter_num);
                    nmii = zeros(1,iter_num);
                    avgenti = zeros(1,iter_num);
                    ARi = zeros(1,iter_num);
                    RIi = zeros(1,iter_num);
                    MIi = zeros(1,iter_num);
                    HIi = zeros(1,iter_num);
                    linshi_indi = zeros(1,length(indicator));
                    options.k = para_k;
                    tic;
                    for f = 1:iter_num
                        ind_folds = folds{f};
                        linshi_AAW = 0;
                        linshi_WWW= 0;
                        S_ini = 0;
                        X_queshi=cell(length(X),1);
                        W_graph=cell(length(X),1);
                        G = cell(length(X),1);
                        for iv = 1:length(X)
                            
                            X1 = X{iv};
                            
                            ind_1 = find(ind_folds(:,iv) == 1);

                            ind_0 = find(ind_folds(:,iv) == 0);
                            X1(:,ind_0) = [];   
                            
                            X_queshi{iv} = X1;       
                            % % % %     % ------------- missing-view index ----------- %
                            
                            linshi_G = diag(ind_folds(:,iv));
                            linshi_G(:,ind_0) = [];
                            G{iv} = linshi_G;

                            linshi_W = full(constructW(X1',options))+eye(size(X1,2));
                            W_graph{iv} = (linshi_W+linshi_W')*0.5;
                            
                        end
                        max_iter = 100;
                        [Con_P,obj]=LSIMVC5gpu_s(X_queshi,W_graph,G,ind_folds,numClust,lambda,beta,para_r,max_iter);
                        new_F = double(Con_P');
                        norm_mat = repmat(sqrt(sum(new_F.*new_F,2)),1,size(new_F,2));
                        % avoid divide by zero
                        for i = 1:size(norm_mat,1)
                            if (norm_mat(i,1)==0)
                                norm_mat(i,:) = 1;
                            end
                        end
                        new_F = new_F./norm_mat;
                        
                        pre_labels    = kmeans(new_F,numClust,'emptyaction','singleton','replicates',20,'display','off','Options',statset('UseParallel',1));
                        
                        result_cluster = ClusteringMeasure(truthF, pre_labels)*100;
                        acc(f) = result_cluster(1);
                        nmi(f) = result_cluster(2);
                        pur(f) = result_cluster(3);
                        if size(pre_labels,1)~=size(truthF,1)
                            pre_labels = pre_labels';
                        end
                        [~, nmii(f), avgenti(f)] = compute_nmi(truthF, pre_labels);
                        [Fi(f),Pi(f),Ri(f)] = compute_f(truthF, pre_labels);
                        [ARi(f),RIi(f),MIi(f),HIi(f)]=RandIndex(truthF, pre_labels);
                    end
                    toc;
                    linshi_indi(1) = mean(acc);
                    std_acc  = std(acc);
                    linshi_indi(2) = mean(nmi);
                    std_nmi  = std(nmi);
                    linshi_indi(3) = mean(pur);
                    std_pur  = std(pur);
                    linshi_indi(4) = mean(Fi);
                    linshi_indi(5) = mean(Pi);
                    linshi_indi(6) = mean(Ri);
                    linshi_indi(7) = mean(nmii);
                    linshi_indi(8) = mean(avgenti);
                    linshi_indi(9) = mean(ARi);
                    linshi_indi(10) = mean(RIi);
                    disp([num2str(for_count),' ',num2str(linshi_indi(1)),' ',num2str(para_r),' ',num2str(lambda),' ',num2str(beta),' ',num2str(para_k)])
                    for i = 1:length(indicator)
                        if linshi_indi(i) >= best_indicator(i)
                            best_indicator(i) = linshi_indi(i);
                            
                        end
                    end
                    b=fopen(text_name,'a+');
                    %                     fprintf(b,'\n');
                    fprintf(b,'%d %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f %4.9f\n',for_count,linshi_indi(1),std_acc,linshi_indi(2),std_nmi,linshi_indi(3),linshi_indi(4),linshi_indi(5),linshi_indi(6),linshi_indi(7),linshi_indi(8),linshi_indi(9),linshi_indi(10),std_pur,lambda,beta,para_r,para_k);
%                     fprintf(b,'\n');
                    fclose(b);

                end
            end
        end
    end
end
