dataset = 1;
rootpath = ['./dataset ', num2str(dataset), '/']

categories = {'Control', 'Schizophrenia'};

for i = 1:length(categories)
    c = categories{i};
    datapath = [rootpath, c, '_bp/'];
    if dataset == 1
        savepath = [rootpath, c, '/'];
    else
        savepath = [rootpath, c, '_asr/'];
    end

    if ~isfolder(savepath)
        mkdir(savepath);
    end
    
    fileList = dir(datapath);
    for i = 1:length(fileList)
        if ~fileList(i).isdir
            EEG = pop_biosig([datapath, fileList(i).name]);

            % correct ASR
            EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',16,'WindowCriterion','off','BurstRejection','off','Distance','Euclidian');
            
            EEG = pop_saveset( EEG, 'filename',strrep(fileList(i).name,'edf','set'),'filepath', savepath);
        end
    end
end
