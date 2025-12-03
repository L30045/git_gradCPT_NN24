%% gradCPT ERP Analysis (Mountainscape Only) + Group In/Out Statistics
% Author(s): De'Ja Rogers, Miray Altinkaynak
% Updated: 2025-10-08
%
% Description:
% This script performs subject-level and group-level ERP analysis for the
% Gradual Onset Continuous Performance Task (gradCPT), focusing exclusively 
% on *Mountainscape* trials (i.e., the "Go" or target condition).

%
% Key Updates:
% - FieldTrip-based epoching using cfg.trl (fixed window, no NaN padding)
% - Baseline correction implemented via ft_preprocessing
% - Robust handling of both "run-1" and "run-01" TSV filename formats
% - Optional Automated or Manual trial rejection modes
% - Computation and storage of In/Out-of-Zone (Vigilance Time Course median split) 
%   ERPs at both run and subject levels
% - Group-level paired t-tests (pointwise) comparing In vs Out conditions, 
%   with significant time points highlighted on ERP plots


%% Environment
close all; clearvars; clc
restoredefaultpath
addpath(fullfile(getenv('HOME'),'Downloads','fieldtrip-20250811'))
ft_defaults

%% Path
dataDir  = fullfile(getenv('HOME'),'Desktop','GradCPT');              
figDir   = fullfile(getenv('HOME'),'Desktop','GradCPT','figures_group2','ERPs');
if ~exist(figDir,'dir'); mkdir(figDir); end

subjects = {'sub-670','sub-671','sub-673','sub-695'};
eegRuns  = {'gradCPT1','gradCPT2','gradCPT3'};  
Fs = 500;                                         
trialWindow   = [-0.1 0.85];                      
channelsToPlot = {'fz','cz','pz','oz'};           

allRunERPs = struct(); 
allRunERPs.meta.Fs          = Fs;
allRunERPs.meta.trialWindow = trialWindow;

%% Trial rejection 
rejMethod = questdlg('Select trial rejection method:', ...
                     'Trial Rejection', ...
                     'Manual','Automated','Manual');
if isempty(rejMethod); rejMethod = 'Manual'; end

if strcmpi(rejMethod,'Automated')
    answer = inputdlg({'Enter STD threshold:','Enter VAR threshold:'}, ...
                      'Automated Rejection Thresholds', [1 40], {'20','200'});
    if isempty(answer); answer = {'20','200'}; end
    stdThresh = str2double(answer{1});
    varThresh = str2double(answer{2});
else
    stdThresh = 20;
    varThresh = 200;
end

%% SubjectxRun loop 
for i = 1:numel(subjects)
    subjID = subjects{i};
    fprintf('\n--- Processing %s ---\n', subjID);
    subName = matlab.lang.makeValidName(subjID);

    if ~isfield(allRunERPs, subName), allRunERPs.(subName) = struct(); end
    subjFigDir = fullfile(figDir, subjID);
    if ~exist(subjFigDir,'dir'); mkdir(subjFigDir); end

    subjDir = fullfile(dataDir, subjID);

    for j = 1:numel(eegRuns)
        eegRunID = eegRuns{j};

        eegFile  = fullfile(subjDir, sprintf('%s_%s_data_Massref_full.mat', subjID, eegRunID));
        tsvFile1 = fullfile(subjDir, sprintf('%s_gradCPT_run-%d_events.tsv',  subjID, j));  
        tsvFile2 = fullfile(subjDir, sprintf('%s_gradCPT_run-%02d_events.tsv',subjID, j));  

        if exist(tsvFile1,'file')
            tsvFile = tsvFile1;
        elseif exist(tsvFile2,'file')
            tsvFile = tsvFile2;
        else
            warning('TSV not found (run-%d or run-%02d): %s', j, j, fullfile(subjDir,'<...>'));
            continue
        end
        if ~isfile(eegFile)
            warning('EEG not found: %s', eegFile); 
            continue
        end

        % EEG
        a = load(eegFile);
        varNames = fieldnames(a);
        data = a.(varNames{1});
        if ~isfield(data,'trial') || isempty(data.trial)
            warning('No trial in EEG file: %s', eegFile);
            continue
        end

        % FieldTrip format
        ftData = [];
        ftData.trial = data.trial;
        ftData.time  = data.time;
        ftData.label = data.label;

        % Read TSV file
        tsvData = readtable(tsvFile,'FileType','text','Delimiter','\t');
        onsets        = tsvData{:,1};   
        response_code = tsvData{:,7};   
        vtc           = tsvData{:,8};   
        vtcMedian     = median(vtc,'omitnan');

        % BP
        cfg = [];
        cfg.hpfilter = 'yes';
        cfg.hpfreq   = 0.3;    
        cfg.lpfilter = 'yes';
        cfg.lpfreq   = 30;     
        cfg.demean   = 'yes';
        ftData = ft_preprocessing(cfg, ftData);

        preSamp  = round(abs(trialWindow(1))*Fs);
        postSamp = round(trialWindow(2)*Fs);
        onsets_samples = round(onsets(:)*Fs);

        trl = [ onsets_samples - preSamp, ...
                onsets_samples + postSamp, ...
               -preSamp * ones(size(onsets_samples)) ];

        cfg = [];
        cfg.trl = trl;
        epoched = ft_redefinetrial(cfg, ftData);

        cfg = [];
        cfg.demean = 'yes';
        cfg.baselinewindow = [trialWindow(1) 0]; 
        epoched = ft_preprocessing(cfg, epoched);

        trialTimeVec = epoched.time{1};
        if ~isfield(allRunERPs,'meta') || ~isfield(allRunERPs.meta,'time') || isempty(allRunERPs.meta.time)
            allRunERPs.meta.time = trialTimeVec;
        end

        % Channel mapping
        epLabelsLower = lower(epoched.label);
        wantedLower   = lower(channelsToPlot);
        chanMaskAll   = ismember(epLabelsLower, wantedLower);

        % Correct mountains
        mntTrials = find(response_code == 0);
        if isempty(mntTrials)
            warning('%s %s: No Mountainscape trials.', subjID, eegRunID);
            continue
        end

        %  Trial rejection
        keepTrial = true(numel(channelsToPlot), numel(mntTrials)); 

        if strcmpi(rejMethod,'Automated')
            for tIdx = 1:numel(mntTrials)
                t = mntTrials(tIdx);
                trialDataMat = epoched.trial{t}(chanMaskAll, :);  
                trialStd = mean(std(trialDataMat,0,2),'omitnan');
                trialVar = mean(var(trialDataMat,0,2),'omitnan');
                if trialStd >= stdThresh || trialVar >= varThresh
                    keepTrial(:,tIdx) = false; 
                    fprintf('Trial %d rejected (STD=%.2f, VAR=%.2f)\n', t, trialStd, trialVar);
                end
            end
        else % Manual
            for tIdx = 1:numel(mntTrials)
                t = mntTrials(tIdx);
                colorsK = lines(numel(channelsToPlot));
                fTrial = figure('Name',sprintf('%s | %s | Trial %d/%d',subjID,eegRunID,tIdx,numel(mntTrials)), ...
                                'Visible','on'); hold on
                for ch = 1:numel(channelsToPlot)
                    chanIdx = find(strcmpi(epoched.label, channelsToPlot{ch}),1);
                    plot(trialTimeVec, epoched.trial{t}(chanIdx,:), 'Color',colorsK(ch,:), 'LineWidth',1.2);
                end
                xlabel('Time (s)'); ylabel('Amplitude (\muV)'); grid on
                legend(upper(channelsToPlot),'Location','northeastoutside')
                title(sprintf('%s - %s - Trial %d',subjID,eegRunID,t))

                trialDataMat = epoched.trial{t}(chanMaskAll, :);
                trialStd = mean(std(trialDataMat,0,2),'omitnan');
                trialVar = mean(var(trialDataMat,0,2),'omitnan');

                defaultChoice = 'Yes';
                if trialStd >= 20 || trialVar >= 200, defaultChoice = 'No'; end

                choice = questdlg(sprintf('Keep trial?\nMean STD: %.2f\nMean VAR: %.2f',trialStd,trialVar), ...
                                  'Manual Rejection','Yes','No',defaultChoice);
                switch choice
                    case 'Yes'
                    case 'No'
                        removeScope = questdlg('Remove all channels or specific channel?', ...
                                               'Remove Scope','All','Specific','All');
                        if strcmpi(removeScope,'All')
                            keepTrial(:,tIdx) = false;
                        else
                            [sel, ok] = listdlg('PromptString','Select channel to remove:', ...
                                                'SelectionMode','single', ...
                                                'ListString', upper(channelsToPlot));
                            if ok, keepTrial(sel,tIdx) = false; end
                        end
                end
                close(fTrial)
            end
        end

      

        % Kept trial indices per channel
        mntTrialsKept = cell(1,numel(channelsToPlot));
        for ch = 1:numel(channelsToPlot)
            mntTrialsKept{ch} = mntTrials( keepTrial(ch,:) );
        end

        % Number of rejected trials 
        rejectedCount = zeros(1,numel(channelsToPlot));
        for ch = 1:numel(channelsToPlot)
            rejectedCount(ch) = numel(mntTrials) - numel(mntTrialsKept{ch});
        end
        totalRejected = max(rejectedCount); % en az bir kanalda reddedilen trial sayısı
        fprintf('%s %s: Total rejected trials = %d\n', subjID, eegRunID, totalRejected);
        allRunERPs.(subName).rejected(j)    = totalRejected;
        allRunERPs.(subName).totalTrials(j) = numel(mntTrials);

        %  Average ERP (Mountainscape Only) — run-level
        for ch = 1:numel(channelsToPlot)
            chanLabel = channelsToPlot{ch};
            chanIdx   = find(strcmpi(epoched.label, chanLabel),1);

            keptIdx = mntTrialsKept{ch};
            if isempty(keptIdx)
                yAvg = nan(1, numel(trialTimeVec));
            else
                M = zeros(numel(keptIdx), numel(trialTimeVec));
                for ii = 1:numel(keptIdx)
                    M(ii,:) = epoched.trial{keptIdx(ii)}(chanIdx,:);
                end
                yAvg = mean(M,1,'omitnan');
            end

            % run-level ERP 
            allRunERPs.(subName).ERP.(chanLabel)(j,:) = yAvg;

           
            f = figure('Visible','off'); hold on
            plot(trialTimeVec, yAvg, 'b','LineWidth',1.5);
            xline(0,'k--'); xlabel('Time (s)'); ylabel('Amplitude (\muV)'); grid on
            title(sprintf('%s %s - %s - Avg ERP (Mountainscape)',subjID,eegRunID,upper(chanLabel)))
            saveas(f, fullfile(subjFigDir, sprintf('%s_%s_%s_avg_mnt.png',subjID,eegRunID,lower(chanLabel))));
            close(f)
        end

        %  Average In / Out of Zone (Correct Mountains) — run-level
        for ch = 1:numel(channelsToPlot)
            chanLabel = channelsToPlot{ch};
            chanIdx   = find(strcmpi(epoched.label, chanLabel),1);

            keptTrials = mntTrialsKept{ch};
            inTrials   = intersect(keptTrials, find(vtc <= vtcMedian));
            outTrials  = intersect(keptTrials, find(vtc >  vtcMedian));

            % In-Zone 
            if isempty(inTrials)
                yAvgIn = nan(1,numel(trialTimeVec));
            else
                M = zeros(numel(inTrials), numel(trialTimeVec));
                for ii = 1:numel(inTrials)
                    M(ii,:) = epoched.trial{inTrials(ii)}(chanIdx,:);
                end
                yAvgIn = mean(M,1,'omitnan');
            end
            % Out-Zone
            if isempty(outTrials)
                yAvgOut = nan(1,numel(trialTimeVec));
            else
                M = zeros(numel(outTrials), numel(trialTimeVec));
                for ii = 1:numel(outTrials)
                    M(ii,:) = epoched.trial{outTrials(ii)}(chanIdx,:);
                end
                yAvgOut = mean(M,1,'omitnan');
            end

          
            allRunERPs.(subName).InZone.(chanLabel)(j,:)  = yAvgIn;
            allRunERPs.(subName).OutZone.(chanLabel)(j,:) = yAvgOut;

            f = figure('Visible','off'); hold on
            hIn  = plot(trialTimeVec, yAvgIn,  'b','LineWidth',1.5);
            hOut = plot(trialTimeVec, yAvgOut, 'Color',[1 0.5 0],'LineWidth',1.5);
            xline(0,'k--'); xlabel('Time (s)'); ylabel('Amplitude (\muV)'); grid on
            legend([hIn hOut], {'In the Zone','Out of the Zone'},'Location','northeastoutside')
            title(sprintf('%s %s - %s - Avg In/Out (Mountainscape)',subjID,eegRunID,upper(chanLabel)))
            saveas(f, fullfile(subjFigDir, sprintf('%s_%s_%s_avg_inout_mnt.png',subjID,eegRunID,lower(chanLabel))));
            close(f)
        end

        fprintf('Saved Mountainscape plots for %s %s\n', subjID, eegRunID);
    end

    % ---- Subject-level ERP (3-run average)
    for ch = 1:numel(channelsToPlot)
        chanLabel = channelsToPlot{ch};
        if isfield(allRunERPs.(subName),'ERP') && isfield(allRunERPs.(subName).ERP,chanLabel)
            subjERP = mean(allRunERPs.(subName).ERP.(chanLabel),1,'omitnan');

            f = figure('Visible','off'); hold on
            plot(trialTimeVec, subjERP, 'k','LineWidth',2);
            xline(0,'k--'); xlabel('Time (s)'); ylabel('Amplitude (\muV)'); grid on
            title(sprintf('%s - Subject Avg ERP (%s, 3 runs)',subjID,upper(chanLabel)))
            saveas(f, fullfile(subjFigDir, sprintf('%s_subject_avg_%s.png',subjID,lower(chanLabel))));
            close(f)
        end
    end

    %  Subject-level reject summary
    if isfield(allRunERPs.(subName),'rejected')
        totalRej = sum(allRunERPs.(subName).rejected);
        totalAll = sum(allRunERPs.(subName).totalTrials);
        fprintf('>>> %s overall: %d / %d trials rejected (%.1f%%)\n', ...
            subjID, totalRej, totalAll, 100*totalRej/totalAll);
    end
end

%% GROUP-LEVEL In vs Out of Zone — paired t-test (pointwise)
fprintf('\n= GROUP-LEVEL In vs Out of Zone ERP Comparison =\n');

trialTimeVec = allRunERPs.meta.time;  
alphaVal = 0.05;                      
colors = [0 0 1; 1 0.5 0];            

for ch = 1:numel(channelsToPlot)
    chanLabel = channelsToPlot{ch};

    subjIn  = [];
    subjOut = [];

    for i = 1:numel(subjects)
        subName = matlab.lang.makeValidName(subjects{i});
        if isfield(allRunERPs.(subName),'InZone') && ...
           isfield(allRunERPs.(subName).InZone,chanLabel) && ...
           isfield(allRunERPs.(subName),'OutZone') && ...
           isfield(allRunERPs.(subName).OutZone,chanLabel)
            inERP  = mean(allRunERPs.(subName).InZone.(chanLabel), 1, 'omitnan');  % run-mean
            outERP = mean(allRunERPs.(subName).OutZone.(chanLabel),1, 'omitnan');  % run-mean
            if numel(inERP)==numel(trialTimeVec) && numel(outERP)==numel(trialTimeVec)
                subjIn(end+1,:)  = inERP;  
                subjOut(end+1,:) = outERP; 
        end
    end
    end
    if isempty(subjIn)
        warning('No In/Out data found for %s — skipping.', upper(chanLabel));
        continue
    end

    % diffwave
    meanIn   = mean(subjIn, 1, 'omitnan');
    meanOut  = mean(subjOut,1, 'omitnan');
    diffWave = meanIn - meanOut;

    % paired t-test
    [~, pvals] = ttest(subjIn, subjOut);      % NxT giriş → 1xT p-values
    sigMask = pvals < alphaVal;

    % plot
    f = figure('Color','w','Name',sprintf('Group In/Out - %s',upper(chanLabel))); hold on
    plot(trialTimeVec, meanIn,  'Color',colors(1,:), 'LineWidth',2);
    plot(trialTimeVec, meanOut, 'Color',colors(2,:), 'LineWidth',2);
    plot(trialTimeVec, diffWave,'k--','LineWidth',1.5);
    xline(0,'k--'); grid on
    xlabel('Time (s)'); ylabel('Amplitude (\muV)');
    legend({'In-Zone','Out-Zone','Diff'}, 'Location','northeastoutside')
    title(sprintf('Group-level In vs Out ERP (%s) — pointwise t-test (p<%.2f)', upper(chanLabel), alphaVal))

    % Mark significant time points at bottom line
    ylims = ylim;
    sigTimes = trialTimeVec(sigMask);
    if ~isempty(sigTimes)
        scatter(sigTimes, ylims(1)*ones(size(sigTimes)), 14, 'r', 'filled', 'v');
    end
    fprintf('%s: %d significant timepoints (p<%.2f)\n', upper(chanLabel), sum(sigMask), alphaVal);

    saveas(f, fullfile(figDir, sprintf('group_inout_%s.png', lower(chanLabel))));
    close(f)
end

%% 6) (OPSİYONEL) Topoplot örneği (tek subject/tek run üstünden)
% Eğer layout dosyan hazırsa ve timelock çıkarmak istersen:
% cfg = []; timelock = ft_timelockanalysis(cfg, epoched); % epoched son olanı kullanır
% cfg = [];
% cfg.layout   = 'gradCPTLayout_Official.lay'; % kendi .lay dosyan
% cfg.xlim     = [0.3 0.6];    % ör: 300–600 ms
% cfg.zlim     = 'maxabs';
% cfg.marker   = 'on';
% cfg.comment  = 'xlim';
% cfg.colorbar = 'yes';
% figure; ft_topoplotER(cfg, timelock);
%% GROUP-LEVEL In vs Out average plot 
fprintf('\n=GROUP-LEVEL In vs Out of Zone ERP Averages =\n');

%  (exclude list)
excludeList = struct();
excludeList.FZ = {'sub-670'};
excludeList.OZ = {'sub-673','sub-995'};

if ~isfield(allRunERPs,'meta') || ~isfield(allRunERPs.meta,'time')
    error('Time vector missing: allRunERPs.meta.time');
end

trialTimeVec = allRunERPs.meta.time;
colors = [0 0 1; 1 0.5 0]; 

for ch = 1:numel(channelsToPlot)
    chanLabel = channelsToPlot{ch};
    subjIn = [];
    subjOut = [];

    chanUpper = upper(chanLabel);
    excludedSubs = {};
    if isfield(excludeList, chanUpper)
        excludedSubs = excludeList.(chanUpper);
    end


    for i = 1:numel(subjects)
        subID = subjects{i};

       
        if ismember(subID, excludedSubs)
            fprintf('Skipping %s for %s (bad channel)\n', subID, upper(chanLabel));
            continue
        end

        subName = matlab.lang.makeValidName(subID);
        if isfield(allRunERPs.(subName),'InZone') && ...
           isfield(allRunERPs.(subName).InZone,chanLabel) && ...
           isfield(allRunERPs.(subName),'OutZone') && ...
           isfield(allRunERPs.(subName).OutZone,chanLabel)
            inERP  = mean(allRunERPs.(subName).InZone.(chanLabel), 1, 'omitnan');
            outERP = mean(allRunERPs.(subName).OutZone.(chanLabel), 1, 'omitnan');
            if numel(inERP)==numel(trialTimeVec)
                subjIn(end+1,:)  = inERP; 
                subjOut(end+1,:) = outERP; 
            end
        end
    end

    if isempty(subjIn)
        warning('No In/Out data for %s. Skipping...', upper(chanLabel));
        continue
    end

    % Group Average
    meanIn   = mean(subjIn, 1, 'omitnan');
    meanOut  = mean(subjOut,1, 'omitnan');

  
    figure('Color','w','Name',sprintf('Group Averages - %s',upper(chanLabel))); hold on
    plot(trialTimeVec, meanIn,  'Color',colors(1,:), 'LineWidth',2);
    plot(trialTimeVec, meanOut, 'Color',colors(2,:), 'LineWidth',2);
    xline(0,'k--');
    grid on
    xlabel('Time (s)');
    ylabel('Amplitude (\muV)');
    legend({'In-Zone','Out-Zone'}, 'Location','northeastoutside');
    title(sprintf('Group-level In vs Out ERP (%s)', upper(chanLabel)));

    fprintf('%s: %d subjects included\n', upper(chanLabel), size(subjIn,1));
end

fprintf('Group-level average plots completed (clean two-line version).\n');


