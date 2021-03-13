function ranking = generateGraph(result,rank_technique, seed)

Mat = zeros(size(result, 1));

% iterate over each extracted features in the result variable
for i = 1: size(result, 1)
    text = ['correlation of ', num2str(i)];
    disp(text);
   
   tmp = result(i); 
   
   % data variable has the table with each row has following meaning:
   % 1. variate
   % 2. Time
   % 3. Sigma
   % 4. Scale
   % 5. Octave
   
   data = tmp{1,1};
   
   % check if the features exist for the respective feature in the input
   % data.
   if size(data,2) == 1  || isempty(data)
      continue; 
   end
   
   % newRangeStart and newRangeStart defines start and end of the temporal scope of the extracted features 
   % newRangeStart = Time - (3 x Sigma)
   % newRangeEnd = Time + (3 x Sigma)
   
   newRangeStart = data(2,:)-3*data(3,:);
   newRangeEnd = data(2,:)+3*data(3,:);
   Sig1 = 6*data(3,:);
   
   newRange1 = [newRangeStart ; newRangeEnd ; Sig1];
   res = [];
   k = 1;
   
   % This loop is to find the temporal alignment of the one feature iterate with the other features in the result variable
   for j = 1: size(result, 1)
          if i ~= j
            tmp = result(j); 
            data1 = tmp{1,1};
            if size(data1,2) == 1 || isempty(data1)
              continue; 
            end
            
            newRangeStart = data1(2,:)-3*data1(3,:);
            newRangeEnd = data1(2,:)+3*data1(3,:);
            Sig2 = 6*data1(3,:);

            newRange2 = [newRangeStart ; newRangeEnd ; Sig2];
            
            totalDis = 0;
            totalDis1 = 0;
            bestMatches = [];
            bestMatches1 = [];
            n = 1;
            
            %  The op == 1 used when Q~D 
                
            for newi = 1:size(newRange1, 2)
                matches = [];
                m = 1; 
                for newj = 1:size(newRange2, 2)
                    rangeI = newRange1(:,newi);
                    rangeJ = newRange2(:,newj);

                    % finding the temporal similarity of two extracted features using = min(U1,U2) - max(L1,L2) 
                    dis = min(rangeI(2), rangeJ(2)) - max(rangeI(1), rangeJ(1));

                    % check for the positive temporal scope overlap of
                    % two features if positive then we store the dis in
                    % the list i.e. in matches
                    if dis >0
                    matches(m) = dis;
                    m = m + 1;    
                    end
                end

                % select the best match with maximum temporal scope
                % overlap of Q with D
                best = max(matches);

                % Create a list of best matches for each features in
                % the Q

                if ~isempty(best)
                    bestMatches(n)= best;
                    n = n + 1;
                end
            end

            % summing all the best matches for all the features in Q with features in D
            totalDis = sum(bestMatches);
            % average =   Sum of bestMatches / No of Features in Q 
            Sum_Of_totalDis1 = totalDis / size(newRange1, 2);

            n = 1;

            for newi = 1:size(newRange2, 2)
                matches = [];
                m = 1; 
                for newj = 1:size(newRange1, 2)
                    rangeI = newRange1(:,newj);
                    rangeJ = newRange2(:,newi);

                    % finding the temporal similarity of two extracted features using = min(U1,U2) - max(L1,L2) 
                    dis = min(rangeI(2), rangeJ(2)) - max(rangeI(1), rangeJ(1));

                    % check for the positive temporal scope overlap of
                    % two features if positive then we store the dis variable value in
                    % the list i.e. in matches variable.
                    if dis >0
                    matches(m) = dis;
                    m = m + 1;    
                    end
                end

                % select the best match with maximum temporal scope
                % overlap of D with Q
                best = max(matches);
                if ~isempty(best)
                    bestMatches1(n)= best;
                    n = n + 1;
                end
            end

        % summing all the best matches for all the features in D with
        % features in Q
            totalDis1 = sum(bestMatches1);

            % average =   Sum of bestMatches / No of Features in D 
            Sum_Of_totalDis2 = totalDis1 / size(newRange2, 2);

            Sum_Of_totalDis = Sum_Of_totalDis1 + Sum_Of_totalDis2

            if Sum_Of_totalDis == 0
                continue;
            end

            SimilarityInTwoFeature = Sum_Of_totalDis;
                
            
            % appending the result of Similarity between two different features
            res(k,:) = [i j SimilarityInTwoFeature];
             
            k= k+1;
         end    
   end
   
   % formation of Similarity Matrix 
   for y = 1: size(res,1)
       Mat(res(y,1),res(y,2)) = res(y,3);
   end

   if isempty(res)
       continue;
   end

end

if rank_technique == 'KNN' 
    r = Mat(end,:);
end

%
if rank_technique == 'PPR' && seed ~= -1
    r = PPR(Mat,(seed), 0.85,size(Mat,1));
end

if rank_technique == 'PR'
    r = PR(Mat);
end

try
    [out,idx] = sort(r);
    kgRes = ["",""];
    idxK = 1;
    for i = size(idx,2): -1 :2
        name = result{idx(i),2};
        value = out(i);
        kgRes(idxK,:) = [name value];
        idxK = idxK+1;
    end
    ranking = kgRes(:, 1:1);
catch
    warning('Please provide correct variate ranking technique. If you have selected rank_technique == PPR also specificy seed node');
end

