dataset = 1;
rootpath = ['./dataset ', num2str(dataset), '/'];

categories = {'Control', 'Schizophrenia'};

for ci = 1:length(categories)
    c = categories{ci};
    datapath = [rootpath, c, '_raw/'];
    savepath = ['./dataset1_for_crossvalidation/', c, '/'];

    if ~isfolder(savepath)
        mkdir(savepath);
    end

    fileList = dir(fullfile(datapath, '*.edf'));
    for fi = 1:length(fileList)
        file_name = fileList(fi).name;
        full_path = fullfile(datapath, file_name);
        
        % Step 1: 載入 EDF 檔案
        EEG = pop_biosig(full_path);

        % Step 2: 轉換為 bipolar leads
        bipolar_pairs = {
            'Fp1', 'F3'; 'F3', 'C3'; 'C3', 'P3'; 'P3', 'O1';
            'Fp2', 'F4'; 'F4', 'C4'; 'C4', 'P4'; 'P4', 'O2';
            'Fp1', 'F7'; 'F7', 'T3'; 'T3', 'T5'; 'T5', 'O1';
            'Fp2', 'F8'; 'F8', 'T4'; 'T4', 'T6'; 'T6', 'O2';
            'T3', 'C3'; 'C3', 'Cz'; 'Cz', 'C4'; 'C4', 'T4';
        };

        new_data = [];
        new_labels = {};

        for i = 1:size(bipolar_pairs, 1)
            ch1 = bipolar_pairs{i,1};
            ch2 = bipolar_pairs{i,2};

            idx1 = find(strcmpi({EEG.chanlocs.labels}, ch1));
            idx2 = find(strcmpi({EEG.chanlocs.labels}, ch2));

            if isempty(idx1) || isempty(idx2)
                warning('找不到頻道 %s 或 %s，跳過此組。', ch1, ch2);
                continue;
            end

            bipolar_signal = EEG.data(idx1,:,:) - EEG.data(idx2,:,:);
            new_data(end+1,:,:) = bipolar_signal';
            new_labels{end+1} = [ch1 '-' ch2];
        end

        EEG.data = new_data;
        EEG.nbchan = size(new_data, 1);
        EEG.chanlocs = struct('labels', new_labels);
        EEG.chaninfo = [];
        EEG = eeg_checkset(EEG);

        % Step 3: Bandpass filter 0.5–45 Hz
        EEG = pop_eegfiltnew(EEG, 0.5, 45);
% 
        % Step 4: Downsample to 125 Hz
        EEG = pop_resample(EEG, 125);

        % Step 5: ASR
        EEG = pop_clean_rawdata(EEG, ...
            'FlatlineCriterion','off', ...
            'ChannelCriterion','off', ...
            'LineNoiseCriterion','off', ...
            'Highpass','off', ...
            'BurstCriterion',16, ...
            'WindowCriterion','off', ...
            'BurstRejection','off', ...
            'Distance','Euclidian');

        % Step 6: 儲存為 .set
        EEG = pop_saveset(EEG, ...
            'filename', strrep(file_name, '.edf', '.set'), ...
            'filepath', savepath);
    end
end
