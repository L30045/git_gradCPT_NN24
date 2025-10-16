%% ============================================================
%  gradCPT EEG Preprocessing Pipeline (Simplified Version)
%  ============================================================
%

%
%  Author:  De'Ja Rogers, Updated and simplified by:  Miray Altinkaynak
%
%   updates:
%   - Streamlined structure for reproducibility and automation
%   - Removed redundant figures and prompts
%   - Replaced variable inconsistencies (clean_data used consistently)
%   - Baseline correction removed at continuous stage 
%     (to be applied post-epoching)
%   - ICA inspection simplified for faster component evaluation
%   - Standardized save structure and naming
%

%
%  Dependencies:
%   FieldTrip (latest version)
%   gradCPTLayout_Official.mat (for layout)
%
%  September 2025
%  ============================================================




clc;
clear all;
close all;
restoredefaultpath
addpath(fullfile(getenv('HOME'),'Downloads','fieldtrip-20250811'))
ft_defaults


%% Initializations
dataFolder   = fullfile(getenv('HOME'),'Desktop','GradCPT','sub-695');  
eeg_filename = 'sub-695_gradCPT1.vhdr';  
fileName     = 'sub-695_gradCPT1';

outDir = fullfile(getenv('HOME'),'Desktop','gradCPT_outputs');
if ~exist(outDir,'dir'), mkdir(outDir); end

layout = load('gradCPTLayout_Official.mat').lay;

subjectNumber = '695';
gradCPTNumber = 1;  

%% Preprocessing
cfg             = [];
cfg.dataset     = [dataFolder filesep eeg_filename];
data            = ft_preprocessing(cfg);

% Filtering
cfg              = [];
cfg.demean       = 'yes';
cfg.bpfilter     = 'yes';
cfg.bpfilttype   = 'firws';
cfg.bpfreq       = [0.1 45];
cfg.detrend      = 'yes';
data             = ft_preprocessing(cfg, data);

cfg             = [];
cfg.continuous  = 'yes';
cfg.layout      = layout;
review = ft_databrowser(cfg, data);

% Remove EOG channels temporarily
cfg             = [];
cfg.channel     = {'all','-hEOG','-vEOG','-Trigger'};
data_noeog      = ft_preprocessing(cfg, data);

% Extract vertical EOG
cfg=[]; cfg.channel='vEOG';
eogv = ft_preprocessing(cfg, data);
eogv = ft_selectdata(cfg, eogv);
eogv.label={'vEOG'}; eogv.senstype='EOG';

% Extract horizontal EOG
cfg=[]; cfg.channel='hEOG';
eogh = ft_preprocessing(cfg, data);
eogh = ft_selectdata(cfg, eogh);
eogh.label={'hEOG'}; eogh.senstype='EOG';


data = ft_appenddata([], data_noeog, eogv, eogh);

% Re-reference to mastoids
cfg=[]; 
cfg.channel   = {'all','-eogh','-eogv'};
cfg.reref     = 'yes';
cfg.refchannel= {'tp9h' 'tp10h'};
cfg.implicitref='tp10h';
data = ft_preprocessing(cfg, data);

%% ICA  
tic
cfg_ica=[]; 
cfg_ica.method  = 'runica';
cfg_ica.channel = {'all','-tp10h'};
data_ica = rmfield(ft_componentanalysis(cfg_ica, data),'cfg');
disp(['ICA took ' num2str(toc) ' seconds'])


cfg=[]; cfg.continuous='no'; cfg.viewmode='component'; cfg.layout=layout;
component_rej = ft_databrowser(cfg, data_ica);

prompt  = {'Enter ICs to remove (space-separated)'};
title   = 'ICs to Remove';
dims    = [1 50];
defaults= {' '};
answer  = inputdlg(prompt,title,dims,defaults);
IC2remove = str2num(answer{1}); %#ok<ST2NM>

% ERPimage & power spectrum per IC
for IC = 1:numel(IC2remove)
    tmp_comp = [];
    for i = 1:length(data_ica.trial)
        tmp_comp = [tmp_comp; data_ica.trial{i}(IC2remove(IC),:)];
    end
    ERPimages{IC} = tmp_comp;
end

cfg=[]; cfg.output='pow'; cfg.channel=IC2remove;
cfg.method='mtmfft'; cfg.taper='hanning'; cfg.foi=0:1:45;
freq = ft_freqanalysis(cfg, data_ica);

for IC = 1:length(ERPimages)
    figure; tiledlayout(2,1); nexttile;
    plot(data_ica.trial{1}(IC2remove(IC),:));
    xlabel('time [ms]'); ylabel('Amp [\muV]');
    nexttile; imagesc(ERPimages{IC}); colorbar;
    figure; cfg=[]; cfg.component=IC2remove(IC); cfg.layout=layout;
    ft_topoplotIC(cfg, data_ica);
    a = input('Should this IC be removed (y/n)? ','s');
    if strcmpi(a,'n'); IC2remove(IC)=nan; end
end
IC2remove = IC2remove(~isnan(IC2remove));

% Remove components
cfg=[]; cfg.component=IC2remove;
clean_data = ft_rejectcomponent(cfg, data_ica);
clean_data.rejected_ICs = IC2remove;

%%  Channel rejection
cfg=[]; cfg.method='summary'; cfg.channel={'fz','cz','pz','oz'}; cfg.ylim=[-50 50];
clean_temp_data = ft_rejectvisual(cfg, clean_data);
channels2remove = setdiff(clean_data.label, clean_temp_data.label);

if ~isempty(channels2remove)
    ind2remove = zeros(1,numel(channels2remove));
    for i=1:numel(channels2remove)
        ind2remove(i)=find(strcmp(clean_data.label,channels2remove{i}));
    end
    temp_labels = clean_data.label; temp_labels(ind2remove)=[];
    cfg=[]; cfg.channel=temp_labels;
    clean_data = ft_selectdata(cfg, clean_data);
end
clean_data.rejected_channels = channels2remove;

%%  Channel interpolation
lap_data = clean_data;
cfg=[]; cfg.method='spline'; cfg.layout=layout; cfg.trials='all';
lap_data = ft_scalpcurrentdensity(cfg, lap_data);
elecinfo = lap_data.elec;

if ~isempty(channels2remove)
    cfg_neighb=[]; cfg_neighb.method='triangulation'; cfg_neighb.layout=layout;
    neighbours = ft_prepare_neighbours(cfg_neighb);
    cfg=[]; cfg.method='average'; cfg.layout=layout; 
    cfg.missingchannel=channels2remove; cfg.neighbours=neighbours; cfg.senstype='eeg';
    clean_data = ft_channelrepair(cfg, clean_data);
end

%%  Save
tokens = regexp(fileName, 'sub-(\d{3})_gradCPT(\d)', 'tokens');
subjectNum = tokens{1}{1}; gradCPTNum = tokens{1}{2};
saveFileName = fullfile(outDir, sprintf('sub-%s_gradCPT%s_data_Massref_full.mat', subjectNum, gradCPTNum));
save(saveFileName, 'clean_data','-v7.3');
fprintf('Saved: %s\n', saveFileName);
