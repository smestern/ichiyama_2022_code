% Filter opto stimulation by light duration and intensity
% Aoi Ichiyama, Inoue Lab, 2022.

%% Create structure with epoc onsets for chosen unit

data = struct;

%% Opto Stimulation Epocs

temp_T5ms_0mA = temp_opto_epocs(unit,1);                % duration 5 ms, 0 mA
temp_T5ms_100mA = temp_opto_epocs(unit,2);              % duration 5 ms, 100 mA
temp_T5ms_200mA = temp_opto_epocs(unit,3);              % duration 5 ms, 200 mA
temp_T5ms_300mA = temp_opto_epocs(unit,4);              % duration 5 ms, 300 mA
temp_T10ms_0mA = temp_opto_epocs(unit,5);               % duration 10 ms, 0 mA
temp_T10ms_100mA = temp_opto_epocs(unit,6);             % duration 10 ms, 100 mA
temp_T10ms_200mA = temp_opto_epocs(unit,7);             % duration 10 ms, 200 mA
temp_T10ms_300mA = temp_opto_epocs(unit,8);             % duration 10 ms, 300 mA
temp_T50ms_0mA = temp_opto_epocs(unit,9);               % duration 50 ms, 0 mA
temp_T50ms_100mA = temp_opto_epocs(unit,10);            % duration 50 ms, 100 mA
temp_T50ms_200mA = temp_opto_epocs(unit,11);            % duration 50 ms, 200 mA
temp_T50ms_300mA = temp_opto_epocs(unit,12);            % duration 50 ms, 300 mA

data.opto.trials.T5.P0 = str2num(temp_T5ms_0mA{1});
data.opto.trials.T5.P100 = str2num(temp_T5ms_100mA{1});
data.opto.trials.T5.P200 = str2num(temp_T5ms_200mA{1});
data.opto.trials.T5.P300 = str2num(temp_T5ms_300mA{1});
data.opto.trials.T10.P0 = str2num(temp_T10ms_0mA{1});
data.opto.trials.T10.P100 = str2num(temp_T10ms_100mA{1});
data.opto.trials.T10.P200 = str2num(temp_T10ms_200mA{1});
data.opto.trials.T10.P300 = str2num(temp_T10ms_300mA{1});
data.opto.trials.T50.P0 = str2num(temp_T50ms_0mA{1});
data.opto.trials.T50.P100 = str2num(temp_T50ms_100mA{1});
data.opto.trials.T50.P200 = str2num(temp_T50ms_200mA{1});
data.opto.trials.T50.P300 = str2num(temp_T50ms_300mA{1});

if data.opto.trials.T5.P0 > 0
    data.opto.onsets.T5.P0 = temp_opto_onsets(unit,data.opto.trials.T5.P0);
end

if data.opto.trials.T5.P100 > 0
    data.opto.onsets.T5.P100 = temp_opto_onsets(unit,data.opto.trials.T5.P100);
end

if data.opto.trials.T5.P200 > 0
    data.opto.onsets.T5.P200 = temp_opto_onsets(unit,data.opto.trials.T5.P200);
end

if data.opto.trials.T5.P300 > 0
    data.opto.onsets.T5.P300 = temp_opto_onsets(unit,data.opto.trials.T5.P300);
end

if data.opto.trials.T10.P0 > 0
    data.opto.onsets.T10.P0 = temp_opto_onsets(unit,data.opto.trials.T10.P0);
end

if data.opto.trials.T10.P100 > 0
    data.opto.onsets.T10.P100 = temp_opto_onsets(unit,data.opto.trials.T10.P100);
end

if data.opto.trials.T10.P200 > 0
    data.opto.onsets.T10.P200 = temp_opto_onsets(unit,data.opto.trials.T10.P200);
end

if data.opto.trials.T10.P300 > 0
    data.opto.onsets.T10.P300 = temp_opto_onsets(unit,data.opto.trials.T10.P300);
end

if data.opto.trials.T50.P0 > 0
    data.opto.onsets.T50.P0 = temp_opto_onsets(unit,data.opto.trials.T50.P0);
end

if data.opto.trials.T50.P100 > 0
    data.opto.onsets.T50.P100 = temp_opto_onsets(unit,data.opto.trials.T50.P100);
end

if data.opto.trials.T50.P200 > 0
    data.opto.onsets.T50.P200 = temp_opto_onsets(unit,data.opto.trials.T50.P200);
end

if data.opto.trials.T50.P300 > 0
    data.opto.onsets.T50.P300 = temp_opto_onsets(unit,data.opto.trials.T50.P300);
end