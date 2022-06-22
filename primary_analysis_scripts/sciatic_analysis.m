% Sciatic nerve stimulation analysis script.

% For analysis in Ichiyama A, Mestern S, et al., 2022. eLife.
% Aoi Ichiyama, Inoue Lab, 2022.

% Supporting script: recording_epocs_NS (to filter stimulation onsets)

% All time in seconds.
% Firing rates in spikes/second or bursts/second.

% "param"                   for input parameters
% "NSparam"                 for sciatic nerve stim (NS) input parameters
% "NS_meanFiringRate"       for sciatic nerve stim binned firing rate
% "NSraster"                for sciatic nerve stim raster plots

clear; clc; close;

%% Import data

% Import spike times (spike times, unit)
temp_spikes = readmatrix('');  
% Import NS onsets and epocs (unit, NS onsets and epocs)
temp_NS = readcell('') ;
temp_NS_epocs = temp_NS(2:end,2:3) ;        % columns with duration/intensity; change as necessary, adjust recording_epocs_NS
temp_NS_onsets = temp_NS(2:end,4:end) ;     % columns with stim onsets; change as necessary, adjust recording_epocs_NS

%% Pattern and Baseline Parameters

param = struct('units',[],'numUnits',[],'burstThres_start',[],'burstThres_fin',[],'burstThres_pre',[]) ;

% Choose units
% temp_unitPrompt = "Unit ID# ";                      % prompt user to choose units to analyze
% param.units = input(temp_unitPrompt) ;              % input single ID or multiple as an array
param.units = [1:36] ;                                 % single ID or multiple IDs as an array
param.numUnits = size(param.units,2) ;              % count number of units

% Choose burst thresholds
param.burstThres_start = 0.006 ;                    % threshold start of burst (sec)
param.burstThres_fin = 0.02 ;                       % threshold end of burst (sec)
param.burstThres_pre = 0.025   ;                    % preburst silence (sec)

%% Sciatic Nerve Stim Parameters

NSparam = struct('NS_binSize',[],'NS_startWindow',[],'NS_finWindow',[],'NS_startBin',[],'NS_finBin',[],'NS_numBins',[],'STIM',[],'REF',[],'SIDE',[]) ;

NSparam.NS_binSize = 1.0 ;                  % bin size (sec);
NSparam.NS_startWindow = -5 ;               % start of first bin (sec)
NSparam.NS_finWindow = 20.0 ;               % end of last bin (sec)

NSparam.NS_startBin = NSparam.NS_startWindow:NSparam.NS_binSize:(NSparam.NS_finWindow - NSparam.NS_binSize) ;
NSparam.NS_finBin = (NSparam.NS_startWindow + NSparam.NS_binSize):NSparam.NS_binSize:NSparam.NS_finWindow ;
NSparam.NS_numBins = length(NSparam.NS_startBin) ;

NSparam.STIM = 'NS' ;
NSparam.REF = 'onsets' ;
NSparam.SIDE = 'contra' ;                   % ipsilateral or contralateral stimulation 'ipsi' or 'contra'

%% Sciatic Nerve Stim Storage

NS_meanFiringRate = struct('NS_all',[],'NS_single',[],'NS_burst',[],'NS_burstEp',[]) ;

% NS response all spikes
NS_meanFiringRate.NS_all = NaN(param.numUnits,NSparam.NS_numBins+1) ;
NS_meanFiringRate.NS_all(2:param.numUnits+1,1) = param.units' ;
NS_meanFiringRate.NS_all(1,2:NSparam.NS_numBins+1) = NSparam.NS_finBin ;
% NS response single spikes
NS_meanFiringRate.NS_single = NaN(param.numUnits,NSparam.NS_numBins+1) ;
NS_meanFiringRate.NS_single(2:param.numUnits+1,1) = param.units' ;
NS_meanFiringRate.NS_single(1,2:NSparam.NS_numBins+1) = NSparam.NS_finBin ;
% NS response burst spikes
NS_meanFiringRate.NS_burst = NaN(param.numUnits,NSparam.NS_numBins+1) ;
NS_meanFiringRate.NS_burst(2:param.numUnits+1,1) = param.units' ;
NS_meanFiringRate.NS_burst(1,2:NSparam.NS_numBins+1) = NSparam.NS_finBin ;
% NS response burst episodes
NS_meanFiringRate.NS_burstEp = NaN(param.numUnits,NSparam.NS_numBins+1) ;
NS_meanFiringRate.NS_burstEp(2:param.numUnits+1,1) = param.units' ;
NS_meanFiringRate.NS_burstEp(1,2:NSparam.NS_numBins+1) = NSparam.NS_finBin ;

%% Sciatic Nerve Stim Plot Storage

NSraster = struct('NS_all',[],'NS_single',[],'NS_burst',[],'NS_burstEp',[]) ;

% Initialize NaN matrix
maxSpikes = (abs(NSparam.NS_startWindow) + abs(NSparam.NS_finWindow)) * 50 * 100 ;   % max estimated as 30 trials and 100 Hz

NSraster.NS_all = struct('all_time',[],'all_trial',[]) ;
NSraster.NS_all.all_time = NaN(maxSpikes,param.numUnits) ;                           % NS all spike times
NSraster.NS_all.all_trial = NaN(maxSpikes,param.numUnits) ;                          % NS all spike trials

NSraster.NS_single = struct('single_time',[],'single_trial',[]) ;
NSraster.NS_single.single_time = NaN(maxSpikes,param.numUnits) ;                     % NS single spike times
NSraster.NS_single.single_trial = NaN(maxSpikes,param.numUnits) ;                    % NS single spike trials

NSraster.NS_burst = struct('burst_time',[],'burst_trial',[]) ;
NSraster.NS_burst.burst_time = NaN(maxSpikes,param.numUnits) ;                       % NS burst spike times
NSraster.NS_burst.burst_trial = NaN(maxSpikes,param.numUnits) ;                      % NS burst spike trials

NSraster.NS_burstEp = struct('burstEp_time',[],'burstEp_trial',[]) ;
NSraster.NS_burstEp.burstEp_time = NaN(maxSpikes,param.numUnits) ;                   % NS burst episode times
NSraster.NS_burstEp.burstEp_trial = NaN(maxSpikes,param.numUnits) ;                  % NS burst episide trials

n = 1 ;

for unit = param.units 
    
    % Timestamps and ISIs for current unit
    TS = temp_spikes(:,unit);
    ISI = diff(TS) ;
    numISI = length(ISI) ;
    TS = TS(1:end-1) ;
    
    % Find burst and single spikes
    burstIDX = zeros(numISI,6) ;
    for i = 1:numISI 
        if ISI(i) <= param.burstThres_start
            burstIDX(i,1) = 1 ;
            if i+1 < numISI
                if ISI(i+1) <= param.burstThres_fin
                    burstIDX(i+1,1) = 1 ;
                end
            end
        end
    end
    
    for j = 2:numISI - 1
        if burstIDX(j,1) == 1 
            if burstIDX(j-1,1) == 0
                burstIDX(j,2) = 1 ;                 % first burst spike
                if ISI(j-1) >= param.burstThres_pre
                    burstIDX(j,4) = 1 ;             % burstEp silence
                end
            end
            if burstIDX(j+1,1) == 0
                burstIDX(j+1,3) = 1 ;               % last burst spike
            end
        end  
    end
 
    if burstIDX(1,1) == 1                           % first spike in array
        burstIDX(1,1:3) = 0 ;
        burstIDX(2,2) = 1 ;
    end
    
    if burstIDX(end,1) == 1                         % last spike in array
        burstIDX(end,3) = 1 ;
    end
    
    for m = 1:numISI-10
            if burstIDX(m,4) == 1
                fill = find(burstIDX(m:m+10,3) == 1, 1) - 1 ;
                burstIDX((m:m+fill),5) = 1 ;
                burstIDX(m+fill,6) = 1 ;
            end
    end

%     preBurstidx = burstEpidx - 1 ;
%     preBurstISI = ISI(preBurstidx) ;
    
    % Firing silence burst and everything else
    singleidx = find(burstIDX(:,5) == 0) ;
    single = TS(singleidx) ;
 
    burstidx = find(burstIDX(:,5) == 1) ;              % bursts with silent period
    burst = TS(burstidx) ;
    
    burstEpidx = find(burstIDX(:,4) == 1) ;            % burst episodes with silent period
    burstEp = TS(burstEpidx) ;
    
    burstEpEndidx = find(burstIDX(:,6) == 1) ;         % burst episodes with silent period end
    burstEpEnd = TS(burstEpEndidx) ;
    
%% Sciatic Nerve Stim
    
    run recording_epocs_NS   
    ONSETS = cell2mat(data.(NSparam.STIM).(NSparam.REF).(NSparam.SIDE));
    numTRIALS = length(ONSETS);
    
    for bin = 1:NSparam.NS_numBins
        trange = [NSparam.NS_startBin(bin) NSparam.NS_finBin(bin)] ;                             % time window for analysis
        tRange = zeros(2,numTRIALS) ;
        tRange(1,:) = ONSETS + trange(1) ;
        tRange(2,:) = ONSETS + trange(2) ;
           
        % Initialize arrays
        all_bin = cell(numTRIALS, 1) ;
        all_count = zeros(numTRIALS, 1) ;
        all_freq = zeros(numTRIALS, 1) ;
        
        single_bin = cell(numTRIALS, 1) ;
        single_count = zeros(numTRIALS, 1) ;
        single_freq = zeros(numTRIALS, 1) ;
        
        burst_bin = cell(numTRIALS, 1) ;
        burst_count = zeros(numTRIALS, 1) ;
        burst_freq = zeros(numTRIALS, 1) ;
        
        burstEp_bin = cell(numTRIALS, 1) ;
        burstEp_count = zeros(numTRIALS, 1) ;
        burstEp_freq = zeros(numTRIALS, 1) ;
        
        for trial = 1:numTRIALS
            trial_all = TS(TS >= tRange(1,trial) & TS < tRange(2,trial)) ;
            all_bin{trial} = trial_all - ONSETS(1,trial) ;
            all_count(trial,1) = cellfun('length', all_bin(trial,1)) ;
            all_freq(trial,1) = all_count(trial,1) / NSparam.NS_binSize ;
            
            trial_single = single(single >= tRange(1,trial) & single < tRange(2,trial)) ;
            single_bin{trial} = trial_single - ONSETS(1,trial) ;
            single_count(trial,1) = cellfun('length', single_bin(trial,1)) ;
            single_freq(trial,1) = single_count(trial,1) / NSparam.NS_binSize ;
            
            trial_burst = burst(burst >= tRange(1,trial) & burst < tRange(2,trial)) ;
            burst_bin{trial} = trial_burst - ONSETS(1,trial) ;
            burst_count(trial,1) = cellfun('length', burst_bin(trial,1)) ;
            burst_freq(trial,1) = burst_count(trial,1) / NSparam.NS_binSize ;
            
            trial_burstEp = burstEp(burstEp >= tRange(1,trial) & burstEp < tRange(2,trial)) ;
            burstEp_bin{trial} = trial_burstEp - ONSETS(1,trial) ;
            burstEp_count(trial,1) = cellfun('length', burstEp_bin(trial,1)) ;
            burstEp_freq(trial,1) = burstEp_count(trial,1) / NSparam.NS_binSize ;
        end
        
        NS_meanFiringRate.NS_all(n+1,bin+1) = mean(all_freq) ;
        NS_meanFiringRate.NS_single(n+1,bin+1) = mean(single_freq) ;
        NS_meanFiringRate.NS_burst(n+1,bin+1) = mean(burst_freq) ;
        NS_meanFiringRate.NS_burstEp(n+1,bin+1) = mean(burstEp_freq) ;
          
    end
    
%% Sciatic Nerve Stim Raster
     
    rasterRange = zeros(2,numTRIALS) ;
    rasterRange(1,:) = ONSETS + NSparam.NS_startWindow ;
    rasterRange(2,:) = ONSETS + NSparam.NS_finWindow ;
    
    % Initialize raster arrays
    allX = cell(numTRIALS, 1) ;
    allY = cell(numTRIALS, 1) ;
    singleX = cell(numTRIALS, 1) ;
    singleY = cell(numTRIALS, 1) ;
    burstX = cell(numTRIALS, 1) ;
    burstY = cell(numTRIALS, 1) ;
    burstEpX = cell(numTRIALS, 1) ;
    burstEpY = cell(numTRIALS, 1) ;
    
    for trial = 1:numTRIALS
        trial_all_raster = TS(TS >= rasterRange(1,trial) & TS < rasterRange(2,trial)) ;
        allX{trial} = trial_all_raster - ONSETS(1,trial) ;
        allY{trial} = trial * ones(numel(trial_all_raster), 1) ;

        trial_single_raster = single(single >= rasterRange(1,trial) & single < rasterRange(2,trial)) ;
        singleX{trial} = trial_single_raster - ONSETS(1,trial) ;
        singleY{trial} = trial * ones(numel(trial_single_raster), 1) ;

        trial_burst_raster = burst(burst >= rasterRange(1,trial) & burst < rasterRange(2,trial)) ;
        burstX{trial} = trial_burst_raster - ONSETS(1,trial) ;
        burstY{trial} = trial * ones(numel(trial_burst_raster), 1) ;

        trial_burstEp_raster = burstEp(burstEp >= rasterRange(1,trial) & burstEp < rasterRange(2,trial)) ;
        burstEpX{trial} = trial_burstEp_raster - ONSETS(1,trial) ;
        burstEpY{trial} = trial * ones(numel(trial_burstEp_raster), 1) ;
    end
    
    % Store X (time) and Y (trial #)
    allX(cellfun('isempty',allX)) = [] ;
    X_all = cat(1, allX{:});
    X_all_num = length(X_all) ;
    NSraster.NS_all.all_time(1:X_all_num,n) = X_all ;
    allY(cellfun('isempty',allY)) = [] ;
    Y_all = cat(1, allY{:});
    NSraster.NS_all.all_trial(1:X_all_num,n) = Y_all ;
    
    singleX(cellfun('isempty',singleX)) = [] ;
    X_single = cat(1, singleX{:});
    X_single_num = length(X_single) ;
    NSraster.NS_single.single_time(1:X_single_num,n) = X_single ;
    singleY(cellfun('isempty',singleY)) = [] ;
    Y_single = cat(1, singleY{:});
    NSraster.NS_single.single_trial(1:X_single_num,n) = Y_single ;
    
    burstX(cellfun('isempty',burstX)) = [] ;
    X_burst = cat(1, burstX{:});
    X_burst_num = length(X_burst) ;
    NSraster.NS_burst.burst_time(1:X_burst_num,n) = X_burst ;
    burstY(cellfun('isempty',burstY)) = [] ;
    Y_burst = cat(1, burstY{:});
    NSraster.NS_burst.burst_trial(1:X_burst_num,n) = Y_burst ;
    
    burstEpX(cellfun('isempty',burstEpX)) = [] ;
    X_burstEp = cat(1, burstEpX{:});
    X_burstEp_num = length(X_burstEp) ;
    NSraster.NS_burstEp.burstEp_time(1:X_burstEp_num,n) = X_burstEp ;
    burstEpY(cellfun('isempty',burstEpY)) = [] ;
    Y_burstEp = cat(1, burstEpY{:});
    NSraster.NS_burstEp.burstEp_trial(1:X_burstEp_num,n) = Y_burstEp ;
    
    figure(n)
    subplot(4,1,1)
    scatter(X_all,Y_all,10,'k','.');
    set(gca,'Color', 'None','XLim',[NSparam.NS_startWindow NSparam.NS_finWindow], 'YLim',[0 numTRIALS])
    TITLE = strcat('Unit ',string(param.units(n)),' - All') ;
    title(TITLE) ;
    ylabel('trial (s)')
    xline(0, 'r')
    
    subplot(4,1,2)
    scatter(X_single,Y_single,10,'k','.');
    set(gca,'Color', 'None','XLim',[NSparam.NS_startWindow NSparam.NS_finWindow], 'YLim',[0 numTRIALS])
    TITLE = strcat('Unit ',string(param.units(n)),' - Single') ;
    title(TITLE) ;
    ylabel('trial (s)')
    xline(0, 'r')
    
    subplot(4,1,3)
    scatter(X_burst,Y_burst,10,'k','.');
    set(gca,'Color', 'None','XLim',[NSparam.NS_startWindow NSparam.NS_finWindow], 'YLim',[0 numTRIALS])
    TITLE = strcat('Unit ',string(param.units(n)),' - Burst') ;
    title(TITLE) ;
    ylabel('trial (s)')
    xline(0, 'r')
    
    subplot(4,1,4)
    scatter(X_burstEp,Y_burstEp,10,'k','.');
    set(gca,'Color', 'None','XLim',[NSparam.NS_startWindow NSparam.NS_finWindow], 'YLim',[0 numTRIALS])
    TITLE = strcat('Unit ',string(param.units(n)),' - BurstEp') ;
    title(TITLE) ;
    ylabel('trial (s)')
    xlabel('time (s)')
    xline(0, 'r')
    
    n = n + 1 ;
    
end
     
clear -regexp ^temp ^trial ^X_ ^Y_