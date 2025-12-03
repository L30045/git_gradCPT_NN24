function gradCPT_data2events( fName, subLabel, sesLabel, taskLabel, runIndex )

% Run this in the sourcedata directory with the gradCPT behavioral mat
% file, e.g. sourcedata/raw/sub-subLabel/gradCPT/
% Make sure the code is in the path, and most likely in /code/gradCPT
% This will save the events.tsv file in /sub-subLabel/nirs and a plot of
% the behavioral results in the derivatives folder

% do I worry about my filtfilt being over 10 samples when NaNs can mean
% samples are skipped?

% get folder info
wdo = cd();

subj_path = '';
if ~isempty(subLabel) && ~isempty(taskLabel)
    subj_path = ['sub-' subLabel filesep 'nirs' filesep];
    subj_path0 = ['sub-' subLabel filesep];
    if isempty(sesLabel)
        baseFileNameNoExt = sprintf('sub-%s_task-%s', subLabel, taskLabel );
    else
        baseFileNameNoExt = sprintf('sub-%s_ses-%s_task-%s', subLabel, sesLabel, taskLabel );
    end
    if ~isempty(runIndex)
        baseFileNameNoExt_events = sprintf('%s_run-%s_events', baseFileNameNoExt, runIndex );
        baseFileNameNoExt_eyetracking =  sprintf('%s_run-%s_recording-eyetracking_physio', baseFileNameNoExt, runIndex );
        baseFileNameNoExt = sprintf('%s_run-%s_nirs', baseFileNameNoExt, runIndex );
    else
        baseFileNameNoExt_events = sprintf('%s_events', baseFileNameNoExt);     
        baseFileNameNoExt_eyetracking =  sprintf('%s_recording-eyetracking_physio', baseFileNameNoExt, runIndex );
        baseFileNameNoExt = sprintf('%s_nirs', baseFileNameNoExt );

    end
end

if ~isempty(subj_path)
    if exist(['..' filesep '..' filesep '..' filesep subj_path], 'dir') % ???
        folder = ['..' filesep '..' filesep '..' filesep subj_path];
    elseif exist(['..' filesep '..' filesep '..' filesep 'sourcedata'], 'dir') % e.g. /sourcedata/raw/sub-id
        folder = ['..' filesep '..' filesep '..' filesep subj_path];
        if ~exist( ['..' filesep '..' filesep '..' filesep subj_path0], 'dir' )
            mkdir( ['..' filesep '..' filesep '..' filesep subj_path0] );
        end
        if ~exist( ['..' filesep '..' filesep '..' filesep subj_path], 'dir' )
            mkdir( ['..' filesep '..' filesep '..' filesep subj_path] );
        end
    elseif exist(['..' filesep '..' filesep '..' filesep '..' filesep 'sourcedata'], 'dir') % e.g. /sourcedata/raw/sub-id/gradCPT
        folder = ['..' filesep '..' filesep '..' filesep '..' filesep subj_path];
        folder_plots = ['..' filesep '..' filesep '..' filesep '..' filesep 'derivatives' filesep 'plots' filesep 'gradCPT_performance' filesep];
        if ~exist( ['..' filesep '..' filesep '..' filesep '..' filesep subj_path0], 'dir' )
            mkdir( ['..' filesep '..' filesep '..' filesep '..' filesep subj_path0] );
        end
        if ~exist( ['..' filesep '..' filesep '..' filesep '..' filesep subj_path], 'dir' )
            mkdir( ['..' filesep '..' filesep '..' filesep '..' filesep subj_path] );
        end
        if ~exist(folder_plots, 'dir')
            mkdir(folder_plots)
        end
    end
end


% load a Data file from the matlab psychotoolbox GUI
load(fName);

% first frame onset
frame_start_offset = ttt(1,1) - starttime; % NEW

% fNIRS temporal offset
snirf = SnirfClass([folder baseFileNameNoExt '.snirf']);
idx = find(diff(snirf.aux(1).dataTimeSeries)>0);
t_nirs_offset_sessions = snirf.aux(1).time(idx(1)); % NEW


% CALCULATE VTC
RT = response(1:end-1,5);
t = [1:length(RT)]' * 0.8 + t_nirs_offset_sessions; % FIXME: assumes interval is 0.8 seconds

t_new = ttt(1:end-1,1) - starttime + t_nirs_offset_sessions; %NEW


%     lst_commision_error = find( data(i_ses).response(:,1)==1 & ~isnan(data(i_ses).response(:,5)));
%     lst_omision_error = find( data(i_ses).response(:,1)==2 & isnan(data(i_ses).response(:,5)));
lst_commision_error = find( response(:,1)==1 & response(:,5)~=0 );
lst_omision_error = find( response(:,1)==2 & response(:,5)==0 );

%
% Set up the events variable
%
events = zeros(length(t),7);
events(:,1) = t_new; % NEW onset
events(:,2) = diff(ttt(:,1));% NEW 0.8; % duration
events(:,3) = 1; % amplitude
events(:,4) = response(1:end-1,1); % trial_type
events(:,5) = 0; % WHICH IS IT???? data(i_ses).response(1:end-1,2); % exemplar
events(:,6) = response(1:end-1,5)/1e3; % reaction time
events(:,7) = response(1:end-1,7); % response_code
events(lst_omision_error,7) = -1;
events(lst_commision_error,7) = -2;

events_trial_type = cell(length(t),1);
lst = find(events(:,4)==1); for ii=1:length(lst); events_trial_type{lst(ii)} = 'mnt'; end
lst = find(events(:,4)==2); for ii=1:length(lst); events_trial_type{lst(ii)} = 'city'; end


% Mike's psuedo-code
meanRT = nanmean(RT(:,1));
stdRT = nanstd(RT(:,1),1);

RT(:,2)=RT(:,1);
RT(find(RT(:,2)==0),2)=NaN;

%interp to fill NaNs (or replace with the mean RT)
RT(:,3)=fillmissing(RT(:,2),'previous','endvalues','nearest');
%or alternatively this (but inpaint_nans is not a native matlab function)
%RT(:,3)=inpaint_nans(RT(:,2),4);

RT(:,4)=(((RT(:,3)-meanRT)/stdRT));

RT(:,5)=abs(RT(:,4));

% Smooth the VTC and compute the median
L = 20; 
W=gausswin(L)/2; %creates a gaussian transfer function with width of L (default is 20 trials for us, 
% you can try other sizes or other ways to make this transfer function or smooth the
% time series. Dividing by 2 is just for scaling/visualization, can be included or
% not, as it will not impact results)
VTC_smoothed = filtfilt(W,sum(W),RT(:,5));
median_VTC = median(VTC_smoothed);

% get pupil diameter
%
% PUPIL DIAMETER
% I need the t_pd_offset for the pupil diameter relative to the gradCPT
wdo = cd();
cd( ['../../../../' subj_path] )
if 0 %exist(join([baseFileNameNoExt_eyetracking '.tsv'], ''))
    M = readtable([baseFileNameNoExt_eyetracking '.tsv'],'FileType', 'text', 'Delimiter', '\t');
    pd_left = M.eyeleft_pupilDiameter;
    pd_left = fillmissing(pd_left, 'linear');
    pd_right = M.eyeright_pupilDiameter;
    pd_right = fillmissing(pd_right, 'linear');

    mean_pd = nanmean([pd_right, pd_left], 2);
    t_pd = M.timestamps; %+ t_pd_offset;

    if any(isnan(mean_pd))
        t_pd = t;
        mean_pd = zeros( size(t) );
    end
else
    t_pd = t;
    mean_pd = zeros( size(t) );
end
cd(wdo);


%
% plot the VTC figure
%
hf = figure();
hf.Position = [100, 100, 1200, 800];
[ax,hl1,hl2] = plotyy( t,RT(:,5), t_pd,filtfilt(ones(100,1),100,mean_pd ));
set(hl1,'color',[1 1 1]*0.8,'linewidth',0.25)
set(hl2,'color','m','linewidth',1)
hold on

lst = find( VTC_smoothed < median_VTC );
VTC_smoothed_in = zeros( size(VTC_smoothed) );
VTC_smoothed_in(:) = NaN;
VTC_smoothed_in(lst) = VTC_smoothed(lst);

lst = find( VTC_smoothed >= median_VTC );
VTC_smoothed_out = zeros( size(VTC_smoothed) );
VTC_smoothed_out(:) = NaN;
VTC_smoothed_out(lst) = VTC_smoothed(lst);

hl=plot(ax(1),t,VTC_smoothed_in,'r');
set(hl,'linewidth',2)
hl=plot(ax(1),t,VTC_smoothed_out,'b');
set(hl,'linewidth',2)

% CHECK THE HEIGHT OF THE COMMISSION ERROR MARKERS 
scale = median_VTC + 4*std(VTC_smoothed);
plot( t(lst_omision_error), scale*ones(length(lst_omision_error),1), 'ko', 'markersize', 10, 'MarkerEdgeColor','k','MarkerFaceColor','k' );
plot( t(lst_commision_error), scale*ones(length(lst_commision_error),1), 'kd', 'markersize', 10, 'MarkerEdgeColor','k','MarkerFaceColor',[1 1 1]*0.6 );

lst = find(events(:,4)==1);
yy=ylim();
for i_stim = 1:length(lst)
    plot( [1 1]*t(lst(i_stim),1), yy, 'k');
end

xlim(ax(1),[0 t(end)])
xlim(ax(2),[0 t(end)])

if isempty(lst_commision_error) & isempty(lst_omision_error)
    legend( 'VTC', 'VTC in', 'VTC out' )
elseif isempty(lst_commision_error)
    legend( 'VTC', 'VTC in', 'VTC out', 'Omission' )
elseif isempty(lst_omision_error)
    legend( 'VTC', 'VTC in', 'VTC out', 'Commission' )
else
    legend( 'VTC', 'VTC in', 'VTC out', 'Omission', 'Commission' )
end
set(hf,'color', [1 1 1])

set(ax(1),'fontsize',16)
set(ax(1),'ycolor','k')
ylabel(ax(1),'RT deviance z-score')

set(ax(2),'fontsize',16)
set(ax(2),'ycolor','m')
ylabel(ax(2),'Pupil Diameter (mm)')

xlabel('Time (s)')
fig_title = strrep(baseFileNameNoExt, '_', ' ')
title(fig_title(1:end-5))
hold off

saveas(hf, join([folder_plots, baseFileNameNoExt_events '_VTC.png']))

%
% save the events.tsv file
% add VTC
%
% onset, duration, amplitude, trial_type, exemplar,
%           reaction_time, response_code, VTC
%
fid = fopen( [folder baseFileNameNoExt_events '.tsv'], 'w');
fprintf( fid, 'onset\tduration\tvalue\ttrial_type\texemplar\treaction_time\tresponse_code\tVTC\n');
for ii = 1:length(t)
    fprintf( fid, '%.3f\t%0.3f\t%d\t%s\t%d\t%d\t%d\t%0.3f\n', ...
        events(ii,1), events(ii,2), events(ii,3), events_trial_type{ii}, ...
        events(ii,5), events(ii,6)*1e3, events(ii,7), RT(ii,5) );
end
fclose(fid);

