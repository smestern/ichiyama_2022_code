% Light response analysis script.

% For analysis in Ichiyama A, Mestern S, et al., 2022. eLife.
% Aoi Ichiyama, Inoue Lab, 2022

% Supporting script: recording_epocs_opto (to filter stimulation onsets)

% All time in seconds.
% Firing rates in spikes/second or bursts/second.

% "param"                       for input parameters
% "optoparam"                   for optogenetic stim (opto) input parameters
% "optoMeanFiringRate"          for optogenetic stim binned firing rate
% "optostats"                   for optogenetic stim response prob firing (pre post), t-test, latency
% "optoraster"                  for optogenetic raster plots

clear; clc; close;

%% Import data

% Import spike times (spike times, unit)
temp_spikes = readmatrix('');
% Import opto onsets and epocs (unit, opto onsets and epocs)
temp_opto = readcell('') ;
temp_opto_epocs = temp_opto(2:end,2:13) ;           % columns with duration/intensity; change as necessary, adjust recording_epocs_opto
temp_opto_onsets = temp_opto(2:end,14:end) ;        % columns with stim onsets; change as necessary, adjust recording_epocs_opto

%% Pattern and Baseline Parameters

param = struct('units',[],'numUnits',[],'BL_dur',[],'BL_start',[],'BL_fin',[],'tBinSize',[], 'numBins',[],'burstThres_start',[],'burstThres_fin',[],'burstThres_pre',[]) ;

% Choose units
% temp_unitPrompt = "Unit ID# ";                      % prompt user to choose units to analyze
% param.units = input(temp_unitPrompt) ;              % input single ID or multiple as an array
param.units = [1:36] ;                                 % single ID or multiple IDs as an array
param.numUnits = size(param.units,2) ;              % count number of units

% Choose burst thresholds
param.burstThres_start = 0.006 ;                    % threshold start of burst (sec)
param.burstThres_fin = 0.02 ;                       % threshold end of burst (sec)
param.burstThres_pre = 0.025   ;                    % preburst silence (sec)

%% Opto Stim Parameters

optoparam = struct('opto_binSize',[],'opto_startWindow',[],'opto_finWindow',[],'opto_startBin',[],'opto_finBin',[],'opto_numBins',[],'STIM',[],'REF',[],'DURATION',[],'INTENSITY',[],'opto_preBin',[],'opto_postBin',[]) ;

optoparam.opto_binSize = 0.02 ;                             % bin size (sec);
optoparam.opto_startWindow = -0.02 ;                        % start of first bin (sec)
optoparam.opto_finWindow = 0.1 ;                            % end of last bin (sec)

optoparam.opto_startBin = optoparam.opto_startWindow:optoparam.opto_binSize:(optoparam.opto_finWindow - optoparam.opto_binSize) ;
optoparam.opto_finBin = (optoparam.opto_startWindow + optoparam.opto_binSize):optoparam.opto_binSize:optoparam.opto_finWindow ;
optoparam.opto_numBins = length(optoparam.opto_startBin) ;

optoparam.STIM = 'opto' ;
optoparam.REF = 'onsets' ;
optoparam.DURATION = 'T5' ;         % Light pulse duration: 'T5' or 'T10' or 'T50'
optoparam.INTENSITY = 'P300';       % Light intensity (mA): 'P0' or 'P100' or 'P200' or 'P300'

optoparam.opto_preBin = find(optoparam.opto_finBin == 0) ;
optoparam.opto_postBin = optoparam.opto_preBin + 1 ;

%% Opto Stim Storage

optoMeanFiringRate = struct('opto_all',[],'opto_single',[],'opto_burst',[],'opto_burstEp',[]) ;

% Opto response all spikes
optoMeanFiringRate.opto_all = NaN(param.numUnits,optoparam.opto_numBins+1) ;
optoMeanFiringRate.opto_all(2:param.numUnits+1,1) = param.units' ;
optoMeanFiringRate.opto_all(1,2:optoparam.opto_numBins+1) = optoparam.opto_finBin ;
% Opto response single spikes
optoMeanFiringRate.opto_single = NaN(param.numUnits,optoparam.opto_numBins+1) ;
optoMeanFiringRate.opto_single(2:param.numUnits+1,1) = param.units' ;
optoMeanFiringRate.opto_single(1,2:optoparam.opto_numBins+1) = optoparam.opto_finBin ;
% Opto response burst spikes
optoMeanFiringRate.opto_burst = NaN(param.numUnits,optoparam.opto_numBins+1) ;
optoMeanFiringRate.opto_burst(2:param.numUnits+1,1) = param.units' ;
optoMeanFiringRate.opto_burst(1,2:optoparam.opto_numBins+1) = optoparam.opto_finBin ;
% Opto response burst episodes
optoMeanFiringRate.opto_burstEp = NaN(param.numUnits,optoparam.opto_numBins+1) ;
optoMeanFiringRate.opto_burstEp(2:param.numUnits+1,1) = param.units' ;
optoMeanFiringRate.opto_burstEp(1,2:optoparam.opto_numBins+1) = optoparam.opto_finBin ;

optostats = struct('opto_prob',[],'opto_ttest',[],'opto_latency',[]) ;

optostats.opto_prob = NaN(param.numUnits,3) ;
optostats.opto_prob(:,1) = param.units ;
optostats.opto_ttest = NaN(param.numUnits,3) ;
optostats.opto_ttest(:,1) = param.units ;
optostats.opto_latency = NaN(param.numUnits,3) ;
optostats.opto_latency(:,1) = param.units ;

%% Opto Stim Plot Storage

optoraster = struct('opto_all',[],'opto_single',[],'opto_burst',[],'opto_burstEp',[]) ;

% Initialize NaN matrix
maxSpikes = round((abs(optoparam.opto_startWindow) + abs(optoparam.opto_finWindow)) * 50 * 100) ;   % max estimated as 30 trials and 100 Hz

optoraster.opto_all = struct('all_time',[],'all_trial',[]) ;
optoraster.opto_all.all_time = NaN(maxSpikes,param.numUnits) ;                           % opto all spike times
optoraster.opto_all.all_trial = NaN(maxSpikes,param.numUnits) ;                          % opto all spike trials

optoraster.opto_single = struct('single_time',[],'single_trial',[]) ;
optoraster.opto_single.single_time = NaN(maxSpikes,param.numUnits) ;                     % opto single spike times
optoraster.opto_single.single_trial = NaN(maxSpikes,param.numUnits) ;                    % opto single spike trials

optoraster.opto_burst = struct('burst_time',[],'burst_trial',[]) ;
optoraster.opto_burst.burst_time = NaN(maxSpikes,param.numUnits) ;                       % opto burst spike times
optoraster.opto_burst.burst_trial = NaN(maxSpikes,param.numUnits) ;                      % opto burst spike trials

optoraster.opto_burstEp = struct('burstEp_time',[],'burstEp_trial',[]) ;
optoraster.opto_burstEp.burstEp_time = NaN(maxSpikes,param.numUnits) ;                   % opto burst episode times
optoraster.opto_burstEp.burstEp_trial = NaN(maxSpikes,param.numUnits) ;                  % opto burst episide trials

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

%% Opto Stim
    
    run recording_epocs_opto   
    ONSETS = cell2mat(data.(optoparam.STIM).(optoparam.REF).(optoparam.DURATION).(optoparam.INTENSITY));
    numTRIALS = length(ONSETS);
    
%     % Fix drift from TDT recording system
%     run drift_factor ;
%     for i = 1:numTRIALS
%         ONSETS(i) = ONSETS(i) + (ONSETS(i) * drift) ;
%     end
    
    opto_trials_prepost = zeros(numTRIALS,2) ;

    for bin = 1:optoparam.opto_numBins
        trange = [optoparam.opto_startBin(bin) optoparam.opto_finBin(bin)] ;                             % time window for analysis
        tRange = zeros(2,numTRIALS) ;
        tRange(1,:) = ONSETS + trange(1) ;
        tRange(2,:) = ONSETS + trange(2) ;
           
        % Initialize arrays
        all_bin = cell(numTRIALS, 1) ;
        all_Ybin = cell(numTRIALS, 1) ;
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
            all_Ybin{trial} = trial * ones(numel(all_bin{trial}), 1) ;
            all_count(trial,1) = cellfun('length', all_bin(trial,1)) ;
            all_freq(trial,1) = all_count(trial,1) / optoparam.opto_binSize ;
            
            trial_single = single(single >= tRange(1,trial) & single < tRange(2,trial)) ;
            single_bin{trial} = trial_single - ONSETS(1,trial) ;
            single_count(trial,1) = cellfun('length', single_bin(trial,1)) ;
            single_freq(trial,1) = single_count(trial,1) / optoparam.opto_binSize ;
            
            trial_burst = burst(burst >= tRange(1,trial) & burst < tRange(2,trial)) ;
            burst_bin{trial} = trial_burst - ONSETS(1,trial) ;
            burst_count(trial,1) = cellfun('length', burst_bin(trial,1)) ;
            burst_freq(trial,1) = burst_count(trial,1) / optoparam.opto_binSize ;
            
            trial_burstEp = burstEp(burstEp >= tRange(1,trial) & burstEp < tRange(2,trial)) ;
            burstEp_bin{trial} = trial_burstEp - ONSETS(1,trial) ;
            burstEp_count(trial,1) = cellfun('length', burstEp_bin(trial,1)) ;
            burstEp_freq(trial,1) = burstEp_count(trial,1) / optoparam.opto_binSize ;
        end
        
        optoMeanFiringRate.opto_all(n+1,bin+1) = mean(all_freq) ;
        optoMeanFiringRate.opto_single(n+1,bin+1) = mean(single_freq) ;
        optoMeanFiringRate.opto_burst(n+1,bin+1) = mean(burst_freq) ;
        optoMeanFiringRate.opto_burstEp(n+1,bin+1) = mean(burstEp_freq) ;
        
        all_bin(cellfun('isempty',all_bin)) = [] ;
        X_bin = cat(1, all_bin{:});
        all_Ybin(cellfun('isempty',all_Ybin)) = [] ;
        Y_allbin = cat(1, all_Ybin{:});
        
        if bin == optoparam.opto_preBin
            optostats.opto_prob(n,2) = length(unique(Y_allbin))/numTRIALS ;
            opto_trials_prepost(:,1) = all_count ;
        elseif bin == optoparam.opto_postBin
            optostats.opto_prob(n,3) = length(unique(Y_allbin))/numTRIALS ;
            opto_trials_prepost(:,2) = all_count ;
        end  
    end
    
    % T-test opto response
    [h,p] = ttest(opto_trials_prepost(:,1), opto_trials_prepost(:,2)) ;
    optostats.opto_ttest(n,2) = h ;
    optostats.opto_ttest(n,3) = p ;
    
    % Latency opto response
    firstspike = cellfun(@(v)v(1),all_bin) ;
    optostats.opto_latency(n,2) = mean(firstspike) ;
    optostats.opto_latency(n,3) = std(firstspike) ;
    
%% Opto Stim Raster
     
    rasterRange = zeros(2,numTRIALS) ;
    rasterRange(1,:) = ONSETS + optoparam.opto_startWindow ;
    rasterRange(2,:) = ONSETS + optoparam.opto_finWindow ;
    
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
    optoraster.opto_all.all_time(1:X_all_num,n) = X_all ;
    allY(cellfun('isempty',allY)) = [] ;
    Y_all = cat(1, allY{:});
    optoraster.opto_all.all_trial(1:X_all_num,n) = Y_all ;
    
    singleX(cellfun('isempty',singleX)) = [] ;
    X_single = cat(1, singleX{:});
    X_single_num = length(X_single) ;
    optoraster.opto_single.single_time(1:X_single_num,n) = X_single ;
    singleY(cellfun('isempty',singleY)) = [] ;
    Y_single = cat(1, singleY{:});
    optoraster.opto_single.single_trial(1:X_single_num,n) = Y_single ;
    
    burstX(cellfun('isempty',burstX)) = [] ;
    X_burst = cat(1, burstX{:});
    X_burst_num = length(X_burst) ;
    optoraster.opto_burst.burst_time(1:X_burst_num,n) = X_burst ;
    burstY(cellfun('isempty',burstY)) = [] ;
    Y_burst = cat(1, burstY{:});
    optoraster.opto_burst.burst_trial(1:X_burst_num,n) = Y_burst ;
    
    burstEpX(cellfun('isempty',burstEpX)) = [] ;
    X_burstEp = cat(1, burstEpX{:});
    X_burstEp_num = length(X_burstEp) ;
    optoraster.opto_burstEp.burstEp_time(1:X_burstEp_num,n) = X_burstEp ;
    burstEpY(cellfun('isempty',burstEpY)) = [] ;
    Y_burstEp = cat(1, burstEpY{:});
    optoraster.opto_burstEp.burstEp_trial(1:X_burstEp_num,n) = Y_burstEp ;
    
    figure(n) ;
    subplot(4,1,1);
    scatter(X_all,Y_all,20,'k','.');
    set(gca,'Color', 'None','XLim',[optoparam.opto_startWindow optoparam.opto_finWindow], 'YLim',[0 numTRIALS]) ;
    TITLE = strcat('Unit ',string(param.units(n)),' - All') ;
    title(TITLE) ;
    ylabel('trial (s)') ;
    xline(0, 'r') ;
    
    subplot(4,1,2) ;
    scatter(X_single,Y_single,20,'k','.');
    set(gca,'Color', 'None','XLim',[optoparam.opto_startWindow optoparam.opto_finWindow], 'YLim',[0 numTRIALS]) ;
    TITLE = strcat('Unit ',string(param.units(n)),' - Single') ;
    title(TITLE) ;
    ylabel('trial (s)') ;
    xline(0, 'r') ;
    
    subplot(4,1,3) ;
    scatter(X_burst,Y_burst,20,'k','.');
    set(gca,'Color', 'None','XLim',[optoparam.opto_startWindow optoparam.opto_finWindow], 'YLim',[0 numTRIALS]) ;
    TITLE = strcat('Unit ',string(param.units(n)),' - Burst') ;
    title(TITLE) ;
    ylabel('trial (s)') ;
    xline(0, 'r') ;
    
    subplot(4,1,4) ;
    scatter(X_burstEp,Y_burstEp,20,'k','.');
    set(gca,'Color', 'None','XLim',[optoparam.opto_startWindow optoparam.opto_finWindow], 'YLim',[0 numTRIALS]) ;
    TITLE = strcat('Unit ',string(param.units(n)),' - BurstEp') ;
    title(TITLE) ;
    ylabel('trial (s)') ;
    xlabel('time (s)') ;
    xline(0, 'r') ;
    
    n = n + 1 ;
    
end
     
clear -regexp ^temp ^trial ^X_ ^Y_