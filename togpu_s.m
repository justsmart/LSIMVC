function [outdata] = togpu_s(indata)
% 单精度
%   此处显示详细说明
if isa(indata, 'cell')
    for ind = 1:length(indata)
        indata{ind} = gpuArray(single(full(indata{ind})));
    end
elseif isa(indata, 'numeric')
    indata = gpuArray(single(full(indata)));
else
    try
        indata = gpuArray(single(full(indata)));
    catch
        causeException = MException('MATLAB:gpuArray type of input data error.');
        throw(causeException)
    end
end
outdata = indata;

