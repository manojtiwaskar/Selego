function s = PR(A)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
I = eye(size(A, 1));

% Set a meaningful value for r
r = (0.9 / max(eig(A))); 
y = I - (r * A);

% see Gelfand's formula - convergence of the geometric series of matrices
S = inv(y) - I; 

s = S(end,:);
end

