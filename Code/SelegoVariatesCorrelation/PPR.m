function [r] = PPR(W,seed,c,k)

W=normalize(W, 'range');
ei(k,1)=0;
ei(seed(1),1)=1;
invW=eye(k)-c*W;
invW=inv(invW);
for i=1:size(seed,2)
   ei(seed(i),1)=1/size(seed,2);
end
r=(1-c)*invW;
r=r*ei;

