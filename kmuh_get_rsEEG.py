import numpy as np
import mne
import os
import warnings

import argparse

mne.set_log_level(verbose=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default='./dataset 2/', help='the path contains the final preprocessed folders: Control and Schizophrenia')
    parser.add_argument('--window_length', type=float, default=5.0, help="the sliding window length (second) of each segment when splitting the data")
    parser.add_argument('--window_overlap', type=float, default=4.0, help="the overlap duration between each segment and the previous one")
    parser.add_argument('--sfreq', type=int, default=125, help='sampling rate of data')
    parser.add_argument('--channel', type=int, default=20, help='channel of data')

    args = parser.parse_args()

    def load_timepoints(rootpath):
        timepoint_dic = {}

        f = open(rootpath)
        for line in f.readlines():
            nums = line.replace('\n', '').split(' ')
            if len(nums) == 1:
                subject_id = nums[0]
                timepoint_dic[subject_id] = []
            else:
                timepoint_dic[subject_id].append([int(nums[0]), int(nums[1])])
        f.close

        return timepoint_dic

    for class_name in ['Control', 'Schizophrenia']:
        for dir_name in ['_rsEEG_segment', '']:
            os.makedirs(args.rootpath + class_name + dir_name, exist_ok=True)

        # split data from 1s after the eye-closed event to 1s before the next event (savepath: class_rsEEG_segment)
        timepoint_dic = load_timepoints('kmuh_eye_closed_timepoints.txt')

        file_list = [f for f in os.listdir(args.rootpath + class_name + '_asr') if f.endswith('.set')]
        for file in file_list:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                raw = mne.io.read_raw_eeglab(args.rootpath + class_name + '_asr/' + file)

            subject_id = file.split('.')[0]

            cut = 0
            data = raw.get_data()
            for tps in timepoint_dic[subject_id]:
                rsEEG_segment = data[:args.channel, tps[0]:tps[1]]

                np.save(args.rootpath + class_name + '_rsEEG_segment/' + subject_id.split('-')[0] + "_" + str(cut) + '.npy', rsEEG_segment)
                cut += 1

        # cut each rsEEG segment using a sliding window and merge epochs for each subject
        file_list = os.listdir(args.rootpath + class_name + '_rsEEG_segment/')

        subject_epochs = None
        last_subject = None
        for file in file_list:
            subject_id = file.split('_')[0]

            if subject_id != last_subject:
                if last_subject:
                    if subject_epochs.shape[0] >= 55:
                        np.save(args.rootpath + class_name + '/' + last_subject, subject_epochs)
                subject_epochs = np.array([]).reshape(0, args.channel, int(args.sfreq*args.window_length)) 
                last_subject = subject_id
            
            info = mne.create_info([str(x) for x in range(0, args.channel)], args.sfreq)

            np_data = np.load(args.rootpath + class_name + '_rsEEG_segment/' + file)
            data = mne.io.RawArray(np_data, info)

            if data.get_data().shape[1] >= args.window_length * args.sfreq:
                epochs = mne.make_fixed_length_epochs(data, duration=args.window_length, reject_by_annotation=False, overlap=args.window_overlap)
                subject_epochs = np.concatenate((subject_epochs, epochs.get_data()), axis=0)
        
        if last_subject and subject_epochs.shape[0] >= 55:        # save the last subject
            np.save(args.rootpath + class_name + '/' + last_subject, subject_epochs)

        # rename file name
        class_id = 'h' if class_name == 'Control' else 's'
        file_list = os.listdir(args.rootpath + class_name)
        for num, file in enumerate(file_list):
            os.rename(args.rootpath + class_name + '/' + file, args.rootpath + class_name + '/' + class_id + str(num+1).zfill(2) + '.npy')
