% Filter sciatic nerve stimulation by stimulation side
% Aoi Ichiyama, Inoue Lab, 2022.

%% Create structure with epoc onsets for chosen unit

data = struct;

%% Sciatic Nerve Stim Epocs

temp_NS_ipsi = temp_NS_epocs(unit,1);               % ipsilateral stimulation trials onsets
temp_NS_contra = temp_NS_epocs(unit,2);             % contralateral stimulation trials onsets

data.NS.trials.ipsi = str2num(temp_NS_ipsi{1});
data.NS.trials.contra = str2num(temp_NS_contra{1});

if data.NS.trials.ipsi > 0
    data.NS.onsets.ipsi = temp_NS_onsets(unit,data.NS.trials.ipsi);
end

if data.NS.trials.contra > 0
    data.NS.onsets.contra = temp_NS_onsets(unit,data.NS.trials.contra);
end