function [X,truth,folds] = loadDataset(dataset,datasetFold)
%load dataset

load(datasetFold,"folds");




if contains(datasetFold,"3sources",'IgnoreCase',true)
    load(dataset,"X","truth");
end
if size(X{1},2)~=length(truth)
    for iv = 1:length(X)
        X{iv} = X{iv}';
    end
end
for iv = 1:length(X)
    X{iv} = NormalizeFea(X{iv},0);
end
end

