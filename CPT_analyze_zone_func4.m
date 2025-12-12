%CPT_analyze_zone

function [Output2,MEDIAN_VAR,VARIABILITY_TC]=CPT_analyze_zone_func4(response,data,L,quantile_low,quantile_high,plotYN)

MTN=1;
CITY=2;
CORRECT=1;
ERROR=-1;
NR=0;

if nargin<3 %if FWHM smoothing kernel unspecified, go with 20
    L=20;
end

if nargin<4 %if unspecified use median
    quantile_low=.5;
    quantile_high=.5;
end

if nargin<5 %if unspecified use median
    plotYN=1;
end

%smoothing function
W=gausswin(L);
W = W / sum(W); %normalize

%get RTs    
RT=response(:,5);
% cut last null trial/row
RT=RT(1:length(response)-1,:);

%get all non-zero RTs and mean/std
RT2=RT(find(RT(:,1)>0),:);
meanRT=mean(RT2(:,1));
stdRT=std(RT2(:,1),1);

% indentify no RT trials as NaN    
RT(:,2)=RT(:,1);
RT(find(RT(:,2)==0),2)=NaN;
    
%interp to fill NaNs
RT(:,3)=inpaint_nans(RT(:,2),4);
%RT cols 1=RTs 2=RTs with NaNs 3=RTs with mean replacement

%z tranform RTs and take abs value
RT(:,4)=(((RT(:,3)-meanRT)/stdRT));
RT(:,5)=abs(RT(:,4));  %now col 5 is abs z deviance with mean repl
    
%ORINIGAL VTC
VARIABILITY_TC=filtfilt(W,1,RT(:,5)); %abs 20 this is the 20VTC in trial space
MEDIAN_VAR=median(VARIABILITY_TC);
    
CO_Z=0;
CO_NZ=0;
CE_Z=0;
CE_NZ=0;
CC_Z=0;
CC_NZ=0;
OE_Z=0;
OE_NZ=0;
    
for t=1:length(response)-1
    if response(t,1)==MTN && response(t,7)==NR
        if VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
            CO_Z=CO_Z+1;
        elseif VARIABILITY_TC(t)>=quantile(VARIABILITY_TC,quantile_high)
            CO_NZ=CO_NZ+1;
        end
    elseif response(t,1)==MTN && response(t,7)==ERROR
        if VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
            CE_Z=CE_Z+1;
        elseif VARIABILITY_TC(t)>=quantile(MEDIAN_VAR,quantile_high)
            CE_NZ=CE_NZ+1;
        end
    elseif response(t,1)==CITY && response(t,7)==CORRECT
        if VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
            CC_Z=CC_Z+1;
        elseif VARIABILITY_TC(t)>=quantile(VARIABILITY_TC,quantile_high)
            CC_NZ=CC_NZ+1;
        end
    elseif response(t,1)==CITY && response(t,7)==NR
        if  VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
            OE_Z=OE_Z+1;
        elseif VARIABILITY_TC(t)>=quantile(VARIABILITY_TC,quantile_high)
            OE_NZ=OE_NZ+1;
        end
    end
end

pre_CO_Z=[];
pre_CO_NZ=[];
pre_CE_Z=[];
pre_CE_NZ=[];
CC_Z_RT=[];
CC_NZ_RT=[];
pre_OE_Z=[];
pre_OE_NZ=[];

if response(1,1)==CITY && response(1,7)==CORRECT
    if VARIABILITY_TC(1)<quantile(VARIABILITY_TC,quantile_low)
        CC_Z_RT=[CC_Z_RT; response(1,5)];
    elseif VARIABILITY_TC(1)>=quantile(VARIABILITY_TC,quantile_high)
        CC_NZ_RT=[CC_NZ_RT; response(1,5)];
    end
end

for t=2:length(response)-1
    if response(t-1,1)==CITY && response(t-1,7)==CORRECT
        if response(t,1)==MTN && response(t,7)==NR
            if VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
                pre_CO_Z=[pre_CO_Z; response(t-1,5)];
            elseif VARIABILITY_TC(t)>=quantile(VARIABILITY_TC,quantile_high)
                pre_CO_NZ=[pre_CO_NZ; response(t-1,5)];
            end
        elseif response(t,1)==MTN && response(t,7)==ERROR
            if VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
                pre_CE_Z=[pre_CE_Z; response(t-1,5)];
            elseif VARIABILITY_TC(t)>=quantile(VARIABILITY_TC,quantile_high)
                pre_CE_NZ=[pre_CE_NZ; response(t-1,5)];
            end
        elseif response(t,1)==CITY && response(t,7)==NR
            if  VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
                pre_OE_Z=[pre_OE_Z; response(t-1,5)];
            elseif VARIABILITY_TC(t)>=quantile(VARIABILITY_TC,quantile_high)
                pre_OE_NZ=[pre_OE_NZ; response(t-1,5)];
            end
        end
    end
    if response(t,1)==CITY && response(t,7)==CORRECT
        if VARIABILITY_TC(t)<quantile(VARIABILITY_TC,quantile_low)
            CC_Z_RT=[CC_Z_RT; response(t,5)];
        elseif VARIABILITY_TC(t)>=quantile(VARIABILITY_TC,quantile_high)
            CC_NZ_RT=[CC_NZ_RT; response(t,5)];
        end
    end
end

CE_Z_rate=CE_Z/(CE_Z+CO_Z);
CE_NZ_rate=CE_NZ/(CE_NZ+CO_NZ);
OE_Z_rate=OE_Z/(OE_Z+CC_Z);
OE_NZ_rate=OE_NZ/(OE_NZ+CC_NZ);

pre_CE_Z_RT=mean(pre_CE_Z, 'omitnan');
pre_CE_NZ_RT=mean(pre_CE_NZ, 'omitnan');
pre_OE_Z_RT=mean(pre_OE_Z, 'omitnan');
pre_OE_NZ_RT=mean(pre_OE_NZ, 'omitnan');
pre_CO_Z_RT=mean(pre_CO_Z, 'omitnan');
pre_CO_NZ_RT=mean(pre_CO_NZ, 'omitnan');
CC_Z_RT_mean=mean(CC_Z_RT, 'omitnan');
CC_NZ_RT_mean=mean(CC_NZ_RT, 'omitnan');

CC_SD_Z=std(CC_Z_RT, 'omitnan');
CC_SD_NZ=std(CC_NZ_RT, 'omitnan');


pHit_Z=1-CE_Z/(CE_Z+CO_Z);
pFA_Z=OE_Z/(OE_Z+CC_Z);

pHit_Z(find(pHit_Z(:)==1))=1-.5/(CE_Z+CO_Z);
pHit_Z(find(pHit_Z(:)==0))=.5/(CE_Z+CO_Z);
pFA_Z(find(pFA_Z(:)==0))=.5/(OE_Z+CC_Z);
pFA_Z(find(pFA_Z(:)==1))=1-.5/(OE_Z+CC_Z);
zHIT_Z=norminv(pHit_Z);
zFA_Z=norminv(pFA_Z);
dprime_Z=zHIT_Z-zFA_Z;
criterion_Z=(-1*(zHIT_Z+zFA_Z))/2;

pHit_NZ=1-CE_NZ/(CE_NZ+CO_NZ);
pFA_NZ=OE_NZ/(OE_NZ+CC_NZ);
pHit_NZ(find(pHit_NZ(:)==1))=1-.5/(CE_NZ+CO_NZ);
pHit_NZ(find(pHit_NZ(:)==0))=.5/(CE_NZ+CO_NZ);
pFA_NZ(find(pFA_NZ(:)==0))=.5/(OE_NZ+CC_NZ);
pFA_NZ(find(pFA_NZ(:)==1))=1-.5/(OE_NZ+CC_NZ);
zHIT_NZ=norminv(pHit_NZ);
zFA_NZ=norminv(pFA_NZ);
dprime_NZ=zHIT_NZ-zFA_NZ;
criterion_NZ=(-1*(zHIT_NZ+zFA_NZ))/2;

SDT(1,1)=dprime_Z;
SDT(1,2)=dprime_NZ;
SDT(1,3)=criterion_Z;
SDT(1,4)=criterion_NZ;


Output2=[CE_Z_rate CE_NZ_rate OE_Z_rate OE_NZ_rate CC_Z_RT_mean CC_NZ_RT_mean CC_SD_Z CC_SD_NZ SDT];

%plot code
if plotYN
    figure;
    hold on;
    plot(RT(:,5),'color', [.5 .5 .5]);
    VTC_Z=VARIABILITY_TC;
    VTC_Z(VARIABILITY_TC>=MEDIAN_VAR)=NaN;
    VTC_NZ=VARIABILITY_TC;
    VTC_NZ(VARIABILITY_TC<MEDIAN_VAR)=NaN;
    plot(VTC_Z,'b','LineWidth', 2.5);
    plot(VTC_NZ,'color',[1 .5 0],'LineWidth', 2.5);
    yline(MEDIAN_VAR, 'k','LineStyle', '--','LineWidth', 2);
end

