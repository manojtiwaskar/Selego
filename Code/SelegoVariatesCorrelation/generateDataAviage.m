function result = generateDataAviage(A,octaves, scale, sigma_t)

w = width(A);

result = cell(w,2);


% data variable below will have data format [timestamp x variates] data. One variate
% at a time has to be passed to generateFeature() method for feature
% extraction. 
% Note: Do not pass date/time column of your dataset.

for i = 1:w

    data = table2array(A(:,i));

    if isnumeric(data)
        [frames,gss,dogss] = generateFeature(data, octaves, scale, sigma_t);
        result{i} = frames;
        result{i,2} = A.Properties.VariableNames(i);

    elseif isnumeric(str2double(data))
       data = str2double(data);
       [frames,gss,dogss] = generateFeature(data, octaves, scale, sigma_t);
       result{i} = frames;
       result{i,2} = A.Properties.VariableNames(i);

    else
        data;
    end

end

