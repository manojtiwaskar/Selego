function [path] = getPath2(UG, startNode, step,result)

gSize = size(UG, 1);

path = cell(step,1);

path{1} = result{startNode,2};
res = zeros(5,1);
for i = 1 : step
    idx = find(UG(:,1) == startNode);
    col = 2;
    if isempty(idx)
        idx = find(UG(:,2) == startNode);
        col =1;
    end
    tmp = UG(idx, :);
    tmp = sortrows(tmp,-3);
    k = 1;
    while ~ismember(tmp(k, col), res)
        res(i) = tmp(k,col);
        startNode = res(i);
        path{i+1} = result{startNode,2};
        i =i+1;
        k = k+1;
        if k >= size(tmp,1)
            break;
        end
    end
   
    
end


