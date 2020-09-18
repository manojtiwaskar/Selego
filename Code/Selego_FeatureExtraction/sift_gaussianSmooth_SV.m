function [frames,gss,dogss] = sift_gaussianSmooth_SV(I, Ot, St, sigmaTime, NBP, gthresh, r,sBoundary, eBoundary)
% frames contains:
%
% frames(1) = dependency
% frames(2) = time center
% frames(3) = sigma time
% frames(4) = scale time
% frames(5) = octave time

featureTimeScale = [];
featureDepdScale = [];
% thresh = 0.04 / St / 2 ;
thresh = 0.04 / 3 / 2 ; %value picked from the vivaldi code
NBO    = 8;
NBP_Time = 4;
NBP_Depd = 4;
magnif = 3.0;
% frames      = [] ;
frames = [];
ktime = 2^(1/(St-3));

% Compute scale spaces
% Try this function
stmin=0;%-1;
otmin=0;

[gss,dogss] = gaussianss_time(I, Ot,St,otmin,stmin,St+1, sigmaTime,-1);

for otime = 1: size(gss.octave,2)
    %scaleDiff = St - 1;
    idx = siftlocalmax(  dogss.octave{otime}, 0.8*thresh  ) ;
    idx = [idx , siftlocalmax( - dogss.octave{otime}, 0.8*thresh)] ;  
  
	K=length(idx) ; 
	[i,j,s] = ind2sub( size( dogss.octave{otime} ), idx ) ;
	y=i-1 ;
	x=j-1 ;
	s=s-1+stmin;
    oframes = [x(:)';y(:)';s(:)'] ;
    oframes(4,:) = otime;
    oframes(2,:) = oframes(2,:)*otime;
%    forwardIdx = siftlocalmax_directed_100(dogss.octave{otime}, 0.8*thresh, NormalizeF(depd{odepd}), NormalizeB(depd{odepd}'), scaleDiff);
  tempDepd = oframes(1,:) ;
  tempTime = oframes(2,:) ;
  timeScale = oframes(3,:);
  sigmat =  2^(otime-1+gss.otmin) * gss.sigmat * 2.^(oframes(3,:)/gss.St);
  
  % frames = [frames, [x(:)';y(:)';sigmat(:)';oframes(3,:); oframes(4,:)]];
  frames = [frames, [x(:)'; y(:)'; sigmat(:)'; oframes(3,:); oframes(4,:)]];
end



                                                              
                                                              
                                                              