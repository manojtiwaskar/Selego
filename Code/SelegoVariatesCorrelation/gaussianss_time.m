function [SS,DiffoG] = gaussianss_time(I,Ot,St,ominT,sminT,smaxT,sigmaT0,dsigmaT0)

%I : timeseries column are variate rows are timedata,
%LocM: location matrix
% sigmaNT :  Nominal smoothing of the input timeseries across time
% Ot      :  Numeber of desired octave Time
% St      :  Number of maximum scale over Time
% ominT   :  usually setted to 0 is the minimum octave of time
% sminT   :  minimum  scale over time (it require to compute an offset is <0)
% smaxT   :  minimum  scale over time usually >=3
% sigmaT0 : Smoothing of the level 0 of octave 0 of the scale space. (Note that Lowe's 1.6 value refers to the level -1 of octave 0.)
% dsigmaT0: step between the scale time if ==1 then the scale
% defineStepFactor: it is a flag to define a diferent step factor between different scales.

% Scale multiplicative step
TIMESCALE= zeros(1,4);
timestart1=0;
tic
ktime = 2^(1/St) ;
if sigmaT0 <0.5
    sigmaT0 = 1.6 * ktime ;
end

if dsigmaT0 < 0
    dsigmaT0 = sigmaT0 * sqrt(1 - 1/ktime^2) ; % Scale step factor Time between each scale
end

sigmaNT =0.5;

% Scale space construction
% Save parameters
SS.Ot          = Ot;
SS.St         = St;

SS.sigmat     = sigmaT0;
SS.otmin       = 0;
SS.sminT       = sminT ;
SS.smaxT       = smaxT ;
% starting octave
otcur = 1;
% Data size
[M, N] = size(I);

% Index offset
soT = -sminT+1 ;

SS.octave{otcur} = zeros(M,3,smaxT) ;
% DiffoG.octave{otcur} = zeros(M,smaxT-1);

% From first octave
STimegsigmafor_OT1_OD1 = sqrt((sigmaT0*ktime^sminT)^2  - (sigmaNT/2^ominT)^2);

%  temp = smoothJustTimeSilv(I, STimegsigmafor_OT1_OD1);
[SS.octave{otcur}(:,:,1)] = smoothJustTimeSilv(I, STimegsigmafor_OT1_OD1);
for otact=1: Ot
    if((otact==1))
        SS = Smooth_Asyn(SS, otact, ktime, sminT,smaxT,dsigmaT0,soT);
        DiffoG.octave{otact} = SS.octave{otact}(:,:,1:end-1)-SS.octave{otact}(:,:,2:end);
    else
        sbest_time = min(sminT + St, smaxT) ;
        %half size of time
        TMP= halveSizeTime(squeeze(SS.octave{otact-1}(:,:,sbest_time+soT)));
        target_sigmaT = sigmaT0 * ktime^sminT ;
        prev_sigmaT = sigmaT0 * ktime^(sbest_time - St) ;
        if(target_sigmaT > prev_sigmaT)
            TMP = smoothJustTimeSilv(TMP(:,2), sqrt(target_sigmaT^2 - prev_sigmaT^2)) ;
        end
        SS.octave{otact}(:,:,1)=TMP;
        SS = Smooth_Asyn(SS, otact, ktime, sminT,smaxT,dsigmaT0,soT);
        DiffoG.octave{otact} = SS.octave{otact}(:,:,1:end-1)-SS.octave{otact}(:,:,2:end);
    end
end




    function [SS] = Smooth_Asyn(SS, CurrentTimeOct, ktime,stmin, stmax,sigmaTscaleStep,soT)
        for st=stmin+1:stmax
            dsigmaT =  ktime^(st) * sigmaTscaleStep ;%ktime^(st+1) * sigmaTscaleStep ;
            if st== (stmin)
                % this scale is already computed
            else
                [SS.octave{CurrentTimeOct}(:,:,st+soT)] = smoothJustTimeSilv(squeeze(SS.octave{CurrentTimeOct}(:,2, st+soT-1)),dsigmaT);
            end
        end
    
