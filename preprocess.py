import mne
import os
import warnings
import argparse

mne.set_log_level("WARNING")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=1, help='dataset you want to use', choices=[1, 2])
    parser.add_argument('--rootpath', type=str, default=None, help='the path contains the final preprocessed folders: Control and Schizophrenia (if None, the datapath defaults to "./dataset 1" or "./dataset 2/"; otherwise, the dataset will be set as your input)')
    parser.add_argument('--l_freq', type=float, default=0.5, help='the lower cutoff frequency for bandpass filter')
    parser.add_argument('--h_freq', type=float, default=45.0, help='the upper cutoff frequency for bandpass filter')
    parser.add_argument('--sfreq', type=int, default=125, help='the frequency to downsample to')

    args = parser.parse_args()

    if args.rootpath == None:
        args.rootpath = './dataset ' + str(args.dataset) + '/' 

    filter = [args.l_freq, args.h_freq]

    # channel removal (only dataset 2)
    if args.dataset == 2:
        for c in ['Control', 'Schizophrenia']:
            datapath = args.rootpath + c + '_raw/'
            savepath = args.rootpath + c + '_rm_channel/'
            if not os.path.isdir(savepath):
                os.mkdir(savepath)
            
            # use 20 channel
            remain_channels = ['EEG Fp1-F3', 'EEG F3-C3', 'EEG C3-P3', 'EEG P3-O1', 
                'EEG Fp2-F4', 'EEG F4-C4', 'EEG C4-P4', 'EEG P4-O2', 'EEG Fp1-F7', 
                'EEG F7-T3', 'EEG T3-T5', 'EEG T5-O1', 'EEG Fp2-F8', 'EEG F8-T4', 
                'EEG T4-T6', 'EEG T6-O2', 'EEG T3-C3', 'EEG C3-Cz', 'EEG Cz-C4', 
                'EEG C4-T4']
            
            file_list = os.listdir(datapath)
            for file in file_list:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    raw = mne.io.read_raw_edf(datapath + file)

                _, event_names = mne.events_from_annotations(raw)
                if "3 Hz" in event_names and raw.info['sfreq'] == 125:  # Remove data that do not include the '3 Hz' event and data with a sampling rate different from the other data (S18)
                    if 'EEG LOC-F3' in raw.info['ch_names']:        # Some 'EEG Fp1-F3' will be labeled as 'EEG LOC-F3'; Some 'EEG Fp1-F7' will be labeled as 'LOC-F7' --> change to same name
                        mne.rename_channels(raw.info, {'EEG LOC-F3':'EEG Fp1-F3', 'EEG LOC-F7':'EEG Fp1-F7'}) 

                    # Remove redundant channels
                    raw.pick_channels(remain_channels)      

                    if len(raw.ch_names) != 20:
                        print(file + ' channel is still more than 20')
                    
                    mne.export.export_raw(savepath + file, raw)

    # bandpass filter & downsample 
    for c in ['Control', 'Schizophrenia']:
        if args.dataset == 1:
            datapath = args.rootpath + c + '_raw/'
        else:
            datapath = args.rootpath + c + '_rm_channel/'
        
        savepath = args.rootpath + c + '_bp/'
        if not os.path.isdir(savepath):
                os.mkdir(savepath)

        file_list = os.listdir(datapath)
        for file in file_list:
            raw = mne.io.read_raw_edf(datapath + file, preload=True)
            mne_filter = raw.filter(filter[0], filter[1])   # bandpass
            mne_filter.resample(sfreq=args.sfreq)    # downsample

            mne.export.export_raw(savepath + file, mne_filter)




