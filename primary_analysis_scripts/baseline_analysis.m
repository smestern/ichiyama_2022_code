% Baseline firing rate and firing pattern analysis script.

% For analysis in Ichiyama A, Mestern S, et al., 2022. eLife.
% Aoi Ichiyama, Inoue Lab, 2022.

% All time in seconds.
% Firing rates in spikes/second or bursts/second.

% "param"                   for input parameters
% "BL_meanFiringRate"       for baseline average firing rates (units in rows; columns: all, single, burst, burstEp)
% "BL_tBinS_units"          for baseline running average firing rates (units in columns)
% "BL_tBinS_single_units"   for baseline running average single spike firing rates (units in columns)
% "BL_tBinS_burst_units"    for baseline running average burst spike firing rates (units in columns)
% "BL_tBinS_burstEp_units"  for baseline running average burst episode firing rates (units in columns)
% "BL_tBinSmooth_units"     for baseline smoothed running average firing rates (units in columns)

% "BL_spikesperBurst_units" for baseline spikes per burst (units in columns)
% "BL_meanPrePostEvent"     for baseline ISI length pre and post by event length (units in columns)

% "probPostISI_byPre"       for buszaki pre-predict post ISI probability (units in columns)
% "probPreISI_byPost"       for buszaki post-predict pre ISI probability (units in columns)

clear; clc; close;

%% Import data

% Import spike times (spike times, unit)
temp_spikes = readmatrix('');

%% Pattern and Baseline Parameters

param = struct('units',[],'numUnits',[],'BL_dur',[],'BL_start',[],'BL_fin',[],'tBinSize',[], 'numBins',[],'burstThres_start',[],'burstThres_fin',[],'burstThres_pre',[]) ;

% Choose units
% temp_unitPrompt = "Unit ID# ";                    % prompt user to choose units to analyze
% param.units = input(temp_unitPrompt) ;            % input single ID or multiple as an array
param.units = [1:36] ;                              % single ID or multiple IDs as an array
param.numUnits = size(param.units,2) ;              % count number of units

% Baseline analysis range
param.BL_dur = 600 ;                                % duration of baseline (timestamp sec)
param.BL_start = 0 ;                                % start of baseline window (timestamp sec)
param.BL_fin = param.BL_start + param.BL_dur ;      % end of baseline window
param.tBinSize = 10 ;                               % bin size for running average firing rate analysis (s)
param.numBins = param.BL_dur / param.tBinSize ;     % number of bins for running average

% Burst thresholds
param.burstThres_start = 0.006 ;                    % threshold start of burst (sec)
param.burstThres_fin = 0.02 ;                       % threshold end of burst (sec)
param.burstThres_pre = 0.025   ;                    % preburst silence (sec)

% ISI pre and post probability analysis
param.buzBinSize = 0.01 ;                                 % seconds
param.buzmaxISI = 10 ;                                    % seconds
param.buzBins = param.buzBinSize:param.buzBinSize:param.buzmaxISI ;
param.numbuzBins = length(param.buzBins) ;
param.buzThres = 0.006 ;                                    % threshold for pre-post ISI probability

%% Baseline Storage

% Initialize output arrays
BL_meanFiringRate = NaN(param.numUnits,5) ;                     % firing rates (unit(1), all(2), single(3), burst(4), burstEp (5)
BL_meanFiringRate(:,1) = param.units' ;                         % label rows with unit ID#

maxSpikes = param.BL_dur * 100 ;                                % max estimated as 100 Hz for entire baseline
BL_TS_units = NaN(maxSpikes,param.numUnits) ;                   % baseline timestamps
BL_ISI_units = NaN(maxSpikes,param.numUnits) ;                  % baseline interspike interval (ISI) timestamps
BL_IBI_units = NaN(maxSpikes,param.numUnits) ;                  % baseline interburst interval (IBI) timestamps

BL_spikesperBurst_units = NaN(maxSpikes,param.numUnits) ;       % baseline spikes per burst episode

BL_meanPrePostEvent = struct("pre_ISI", [], "post_ISI",[]) ;    % baseline pre-post ISI length by event length
BL_meanPrePostEvent.pre_ISI = struct("single", [], "burstEp", [], "burstEp2", [], "burstEp3", [], "burstEp4plus", []) ;
BL_meanPrePostEvent.pre_ISI.single = NaN(param.numUnits,2);
BL_meanPrePostEvent.pre_ISI.single(:,1) = param.units' ;
BL_meanPrePostEvent.pre_ISI.burstEp = NaN(param.numUnits,2);
BL_meanPrePostEvent.pre_ISI.burstEp(:,1) = param.units' ;
BL_meanPrePostEvent.pre_ISI.burstEp2 = NaN(param.numUnits,2);
BL_meanPrePostEvent.pre_ISI.burstEp2(:,1) = param.units' ;
BL_meanPrePostEvent.pre_ISI.burstEp3 = NaN(param.numUnits,2);
BL_meanPrePostEvent.pre_ISI.burstEp3(:,1) = param.units' ;
BL_meanPrePostEvent.pre_ISI.burstEp4plus = NaN(param.numUnits,2);
BL_meanPrePostEvent.pre_ISI.burstEp4plus(:,1) = param.units' ;
BL_meanPrePostEvent.post_ISI = struct("single", [], "burstEp", [], "burstEp2", [], "burstEp3", [], "burstEp4plus", []) ;
BL_meanPrePostEvent.post_ISI.single = NaN(param.numUnits,2);
BL_meanPrePostEvent.post_ISI.single(:,1) = param.units' ;
BL_meanPrePostEvent.post_ISI.burstEp = NaN(param.numUnits,2);
BL_meanPrePostEvent.post_ISI.burstEp(:,1) = param.units' ;
BL_meanPrePostEvent.post_ISI.burstEp2 = NaN(param.numUnits,2);
BL_meanPrePostEvent.post_ISI.burstEp2(:,1) = param.units' ;
BL_meanPrePostEvent.post_ISI.burstEp3 = NaN(param.numUnits,2);
BL_meanPrePostEvent.post_ISI.burstEp3(:,1) = param.units' ;
BL_meanPrePostEvent.post_ISI.burstEp4plus = NaN(param.numUnits,2);
BL_meanPrePostEvent.post_ISI.burstEp4plus(:,1) = param.units' ;

BL_tBin_units = NaN(param.numBins,param.numUnits) ;             % baseline all spikes running/binned average
BL_tBin_single_units= NaN(param.numBins,param.numUnits) ;       % baseline single spikes running/binned average
BL_tBin_burst_units= NaN(param.numBins,param.numUnits) ;        % baseline burst spikes running/binned average
BL_tBin_burstEpunits= NaN(param.numBins,param.numUnits) ;       % baseline burst episodes running/binned average
BL_tBinSmooth_units = NaN(param.numBins,param.numUnits) ;       % baseline smoothed running/binned average

probPostISI_byPre = NaN(param.numbuzBins, param.numUnits+1) ;   % buszaki pre-predict post ISI probability
probPostISI_byPre(:,1) = param.buzBins ;                        % label rows with preISI length#
probPreISI_byPost = NaN(param.numbuzBins, param.numUnits+1) ;   % buszaki post-predict pre ISI probability
probPreISI_byPost(:,1) = param.buzBins ;                        % label rows with postISI length#

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
    
    % Firing silence burst and everything else
    singleidx = find(burstIDX(:,5) == 0) ;
    single = TS(singleidx) ;
 
    burstidx = find(burstIDX(:,5) == 1) ;              % bursts with silent period
    burst = TS(burstidx) ;
    
    burstEpidx = find(burstIDX(:,4) == 1) ;            % burst episodes with silent period
    burstEp = TS(burstEpidx) ;
    
    burstEpEndidx = find(burstIDX(:,6) == 1) ;         % burst episodes with silent period end
    burstEpEnd = TS(burstEpEndidx) ;

    %% Baseline
    
    % All spikes baseline
    BL_all_idx = (TS >= param.BL_start & TS <= param.BL_fin) ~= 0 ;                 % baseline index
    BL_all_idx(BL_all_idx == 0) = [] ;                                              % remove non-baseline idx (zeros)
    BL_all_TS = TS(BL_all_idx) ;                                                    % baseline timestamps   
    BL_numTS = length(BL_all_TS) ;                                            
    BL_TS_units(1:BL_numTS,n) = BL_all_TS ;
    % Single spikes baseline
    BL_single_idx = (single >= param.BL_start & single <= param.BL_fin) ~= 0 ;
    BL_single_idx(BL_single_idx == 0) = [] ;
    BL_single_TS = single(BL_single_idx) ;
    numBL_single = length(BL_single_TS) ;
    % Burst spikes baseline
    BL_burst_idx = (burst >= param.BL_start & burst <= param.BL_fin) ~= 0 ;
    BL_burst_idx(BL_burst_idx == 0) = [] ;
    BL_burst_TS = burst(BL_burst_idx) ;
    % Burst episodes start baseline
    BL_burstEp_idx = (burstEp >= param.BL_start & burstEp <= param.BL_fin) ~= 0 ;
    BL_burstEp_idx(BL_burstEp_idx == 0) = [] ;
    BL_burstEp_TS = burstEp(BL_burstEp_idx) ;
    numBL_burstEp = length(BL_burstEp_TS) ;
    
    % Calcualte mean baseline firing rates
    BL_meanFiringRate(n,2) = numel(BL_all_TS) / param.BL_dur ;              % all
    BL_meanFiringRate(n,3) = numel(BL_single_TS) / param.BL_dur ;           % single
    BL_meanFiringRate(n,4) = numel(BL_burst_TS) / param.BL_dur ;            % burst
    BL_meanFiringRate(n,5) = numel(BL_burstEp_TS) / param.BL_dur ;          % burst episodes
    
    % Baseline ISIs
    BL_ISI = diff(BL_all_TS) ;                                              % baseline ISIs
    BL_numISI = length(BL_ISI) ;                                            % number of baseline ISIs
    BL_ISI_units(1:BL_numISI,n) = BL_ISI ;                              % store ISIs for group analysis
    % Baseline IBIs
    BL_IBI = diff(BL_burstEp_TS) ;                                          % baseline IBIs
    BL_numIBI = length(BL_IBI) ;                                            % number of baseline IBIs
    BL_IBI_units(1:BL_numIBI,n) = BL_IBI ;                              % store IBIs for group analysis
    
    % Spikes per burst
    BL_spikesperBurst = burstEpEndidx(1:numBL_burstEp) - burstEpidx(1:numBL_burstEp) + 1 ;       % spikes per burst
    BL_spikesperBurst_units(1:numBL_burstEp,n) = BL_spikesperBurst ;
    
    % Pre and post ISIs by event length
    BL_preSingleidx = singleidx(1:numBL_single) - 1 ;
    BL_preSingleidx(BL_preSingleidx == 0) = [] ;
    BL_preBurstEpidx = burstEpidx(1:numBL_burstEp) - 1 ;
    BL_preBurstEpidx(BL_preBurstEpidx == 0) = [] ;
    BL_2BurstEpidx_idx = find(BL_spikesperBurst == 2);
    BL_pre2BurstEpidx = burstEpidx(BL_2BurstEpidx_idx) - 1 ;         % Only for when BL starts at time zero.
    BL_pre2BurstEpidx(BL_pre2BurstEpidx == 0) = [] ;
    BL_3BurstEpidx_idx = find(BL_spikesperBurst == 3) ;
    BL_pre3BurstEpidx = burstEpidx(BL_3BurstEpidx_idx) - 1 ;
    BL_pre3BurstEpidx(BL_pre3BurstEpidx == 0) = [] ;
    BL_4plusBurstEpidx_idx = find(BL_spikesperBurst > 3);
    BL_pre4plusBurstEpidx = burstEpidx(BL_4plusBurstEpidx_idx) - 1 ;    
    BL_pre4plusBurstEpidx(BL_pre4plusBurstEpidx == 0) = [] ;
    BL_postSingleidx = singleidx(1:numBL_single) ;
    BL_postBurstEpidx = burstEpEndidx(1:numBL_burstEp) ;
    BL_post2BurstEpidx = burstEpEndidx(BL_2BurstEpidx_idx) ;         % Only for when BL starts at time zero.
    BL_post3BurstEpidx_idx = find(BL_spikesperBurst == 3) ;
    BL_post3BurstEpidx = burstEpEndidx(BL_3BurstEpidx_idx) ;
    BL_post4plusBurstEpidx_idx = find(BL_spikesperBurst > 3);
    BL_post4plusBurstEpidx = burstEpEndidx(BL_4plusBurstEpidx_idx) ; 
    
    BL_preSingle_ISI = ISI(BL_preSingleidx) ;
    BL_preBurstEp_ISI = ISI(BL_preBurstEpidx) ;
    BL_pre2BurstEp_ISI = ISI(BL_pre2BurstEpidx) ;
    BL_pre3BurstEp_ISI = ISI(BL_pre3BurstEpidx) ;
    BL_pre4plusBurstEp_ISI = ISI(BL_pre4plusBurstEpidx) ;
    BL_postSingle_ISI = ISI(BL_postSingleidx) ;
    BL_postBurstEp_ISI = ISI(BL_postBurstEpidx) ;
    BL_post2BurstEp_ISI = ISI(BL_post2BurstEpidx) ;
    BL_post3BurstEp_ISI = ISI(BL_post3BurstEpidx) ;
    BL_post4plusBurstEp_ISI = ISI(BL_post4plusBurstEpidx) ;
    
    BL_meanPrePostEvent.pre_ISI.single(n,2) = mean(BL_preSingle_ISI) ; 
    BL_meanPrePostEvent.pre_ISI.burstEp(n,2) = mean(BL_preBurstEp_ISI) ;
    BL_meanPrePostEvent.pre_ISI.burstEp2(n,2) = mean(BL_pre2BurstEp_ISI) ;
    BL_meanPrePostEvent.pre_ISI.burstEp3(n,2) = mean(BL_pre3BurstEp_ISI) ;
    BL_meanPrePostEvent.pre_ISI.burstEp4plus(n,2) = mean(BL_pre4plusBurstEp_ISI) ;
    BL_meanPrePostEvent.post_ISI.single(n,2) = mean(BL_postSingle_ISI) ;
    BL_meanPrePostEvent.post_ISI.burstEp(n,2) = mean(BL_postBurstEp_ISI) ;
    BL_meanPrePostEvent.post_ISI.burstEp2(n,2) = mean(BL_post2BurstEp_ISI) ;
    BL_meanPrePostEvent.post_ISI.burstEp3(n,2) = mean(BL_post3BurstEp_ISI) ;
    BL_meanPrePostEvent.post_ISI.burstEp4plus(n,2) = mean(BL_post4plusBurstEp_ISI) ;    

%% Running Average Firing Rates
    
    % Set boundaries for non-overlapping bins
    tBins = zeros(param.numBins,7) ;
    tBins(:,1) = 0:param.tBinSize:param.BL_dur-param.tBinSize ;
    tBins(:,2) = param.tBinSize:param.tBinSize:param.BL_dur ;
% 
    for tbin = 1:length(tBins)
        tBins(tbin,3) = numel(find(BL_all_TS > tBins(tbin,1) & BL_all_TS < tBins(tbin,2))) / param.tBinSize ;           % all spikes binned
        tBins(tbin,4) = numel(find(BL_single_TS > tBins(tbin,1) & BL_single_TS < tBins(tbin,2))) / param.tBinSize ;     % single spikes binned   
        tBins(tbin,5) = numel(find(BL_burst_TS > tBins(tbin,1) & BL_burst_TS < tBins(tbin,2))) / param.tBinSize ;       % burst spikes binned
        tBins(tbin,6) = numel(find(BL_burstEp_TS > tBins(tbin,1) & BL_burstEp_TS < tBins(tbin,2))) / param.tBinSize  ;  % burst episodes binned
    end
    
    BL_tBin_units(:,n) = tBins(:,3) ;
    BL_tBin_single_units(:,n) = tBins(:,4) ;
    BL_tBin_burst_units(:,n) = tBins(:,5) ;
    BL_tBin_burstEpunits(:,n) = tBins(:,6) ;
    
    tBins(:,7) = smooth(tBins(:,3),5) ;                         % Smoothed running average (all spikes)
    BL_tBinSmooth_units(:,n) = tBins(:,7) ;                     % Store smoothed running average for multiple units
    
    %% ISI Analysis - Pre-predict Post
    
    ISI_pre = BL_ISI(1:end-1) ;
    ISI_post = BL_ISI(2:end) ;

    % Sort and round
    [~,idx] = sort(ISI_pre) ;           % sort just the first column by pre
    sorted_preISI = ISI_pre(idx) ;
    sorted_postISI = ISI_post(idx) ;   % sort the whole matrix using the sort indices

    rounded_sorted_preISI = floor(sorted_preISI/param.buzBinSize) * param.buzBinSize + param.buzBinSize ;

    % Count number of unique bins and entries in each bin
    [uniqPre,idxfirstuniqPre,idxuniqPre] = unique(rounded_sorted_preISI,'rows','legacy') ;

    firstcount = idxfirstuniqPre(1) ;
    count = diff(idxfirstuniqPre) ;
    numper_uniqbuzBin = [firstcount; count] ;                % number of entries per bin
    num_uniqbuzBin = length(uniqPre) ;                       % number of bins

    % Find number of post ISIs less than threshold
    numPost6 = zeros(num_uniqbuzBin,1) ;
    a = 1 ;

    for i = 1:num_uniqbuzBin
        idx = find(idxuniqPre == i) ;
        buzPost = sorted_postISI(idx) ; 
        buzThres_idxbuzPost = find(buzPost < param.buzThres) ;
        numPost6(a,1) = numel(buzThres_idxbuzPost) ;
        a = a + 1 ;
    end
    
    % Remove/do not calculate if there are less than 3 instances in a bin
    smalln = find(numper_uniqbuzBin < 3) ;
    numper_uniqbuzBin(smalln) = [] ;
    numPost6(smalln) = [] ;
    uniqPre(smalln) = [] ;

    % Probability that an ISI of less than 6 follows an ISI of bin length
    fraction = numPost6 ./ numper_uniqbuzBin ;
    
    % Store probabilities
    for i = 1:length(uniqPre)
        matchBin = uniqPre(i) ;
        matchidx = find(probPostISI_byPre(:,1) == matchBin) ;
        probPostISI_byPre(matchidx,n+1) = fraction(i) ;
    end
    
    %% ISI Analysis - Pre-predict Post
    
    % Sort and round
    [~,idx] = sort(ISI_post) ;           % sort just the first column
    sorted_preISI = ISI_pre(idx) ;
    sorted_postISI = ISI_post(idx) ;   % sort the whole matrix using the sort indices
    rounded_sorted_postISI = floor(sorted_postISI/param.buzBinSize) * param.buzBinSize + param.buzBinSize ;

    % Count number of unique bins and entries in each bin
    [uniqPost,idxfirstuniqPost,idxuniqPost] = unique(rounded_sorted_postISI,'rows','legacy') ;

    firstcount = idxfirstuniqPost(1) ;
    count = diff(idxfirstuniqPost) ;
    numper_uniqbuzBin = [firstcount; count] ;                % number of entries per bin
    num_uniqbuzBin = length(uniqPost) ;                       % number of bins

    % Find number of post ISIs less than threshold
    numPre6 = zeros(num_uniqbuzBin,1) ;
    a = 1 ;

    for i = 1:num_uniqbuzBin
        idx = find(idxuniqPost == i) ;
        buzPre = sorted_preISI(idx) ; 
        buzThres_idxbuzPost = find(buzPre < param.buzThres) ;
        numPre6(a,1) = numel(buzThres_idxbuzPost) ;
        a = a + 1 ;
    end
    
    % Remove/do not calculate if there are less than 3 instances in a bin
    smalln = find(numper_uniqbuzBin < 3) ;
    numper_uniqbuzBin(smalln) = [] ;
    numPre6(smalln) = [] ;
    uniqPost(smalln) = [] ;

    % Probability that an ISI of less than 6 follows an ISI of bin length
    fraction = numPre6 ./ numper_uniqbuzBin ;
    
    % Store probabilities
    for i = 1:length(uniqPost)
        matchBin = uniqPost(i) ;
        matchidx = find(probPreISI_byPost(:,1) == matchBin) ;
        probPreISI_byPost(matchidx,n+1) = fraction(i) ;
    end
    
    n = n + 1 ;
    
end
     
clear -regexp ^temp ^trial ^X_ ^Y_