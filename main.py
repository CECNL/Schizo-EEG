from pilot_train import *
from full_train import *
from test import *
from interpretation import *
import argparse
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=1, help='dataset you want to use', choices=[1, 2])
    parser.add_argument('--dataset_path', type=str, default='./dataset 1/', help='the path contains the final preprocessed folders: Control and Schizophrenia')
    parser.add_argument('--savepath', type=str, default='./checkpoints/', help='the folder path for saving the training and testing result')

    parser.add_argument('--pilot_train_epoch', type=int, default=30, help='number of epochs that you want to train in pilot train stage')
    parser.add_argument('--full_train_epoch', type=int, default=100, help='number of epochs that you want to train in full train stage')
    parser.add_argument('--fold', type=int, default=5, help="split data into ? folds")

    parser.add_argument('--model', type=str, default='All', help="choose EEG decoding model (choices = 'All', 'Oh_CNN', 'SzHNN', 'EEGNet', 'SCCNet', 'ShallowConvNet', 'MBszEEGNet')")

    parser.add_argument('--sfreq', type=float, default=125, help='sampling rate of data')
    parser.add_argument('--channel', type=int, default=None, help='channel of data (if None, channels default to 19 for dataset1 and 20 for dataset2; otherwise, the specified integer is used)')
    parser.add_argument('--window_length', type=float, default=5.0, help="the sliding window length (second) of each segment when splitting the data")
    parser.add_argument('--window_overlap', type=float, default=4.0, help="the overlap duration between each segment and the previous one")

    parser.add_argument('--seed', type=int, default=None, help="")

    args = parser.parse_args()

    #### Set seed
    if args.seed is None:
        args.seed = 911 if args.dataset == 1 else 1234

    #### Set channel num.
    if args.channel is None:
        args.channel = 19 if args.dataset == 1 else 20

    #### Subject in each dataset
    if args.dataset == 1:
        hc_subject_num = 14
        sch_subject_num = 14
        xai_used_channel = ['Fp2','F8','T4','T6','O2','Fp1','F7','T3','T5','O1','F4','C4','P4','F3','C3','P3','Fz','Cz','Pz']
        show_channel = xai_used_channel # electrodes that you want to show on the saliency topomap
    else:
        hc_subject_num = 40
        sch_subject_num = 35
        xai_used_channel = ['AF3', 'FC3', 'CP3', 'PO3', 'AF4', 'FC4', 'CP4', 'PO4', 'AF7', 'FT7', 'TP7', 'PO7', 'AF8', 'FT8', 'TP8', 'PO8', 'C5', 'C1', 'C2', 'C6']
        show_channel = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'] 

    #### Parameters used in grid search
    grid_param = {
        'training_batch': [16, 32, 64],     # batch_size
        'training_lr': [0.005, 0.001, 0.0005, 0.0001]  #learning rate
    }

    model_list = ['Oh_CNN', 'SzHNN', 'EEGNet', 'SCCNet', 'ShallowConvNet', 'MBSzEEGNet', 'Conformer'] if args.model == 'All' else [args.model]    

    for model_name in model_list:
        #### Pilot tain
        pilot = Pilot(args, grid_param)
        pilot.load_dataset(hc_subject_num, sch_subject_num)
        pilot.train(model_name)

        #### Full train
        full = Full(args)
        full.get_best_param()
        full.load_dataset(hc_subject_num, sch_subject_num)
        full.train(model_name)
        
        #### Test
        test = Test(args)
        gradient = test.test(model_name, hc_subject_num, sch_subject_num)  # gradient = {0: [subject, epochs, channel, timepoints], 1: [subject, epochs, channel, timepoints]}
        test.load_dataset(hc_subject_num, sch_subject_num)
        test.tsne(model_name)

        #### Interpretation
        xai = Interpretation(args)
        xai.plot_psd(model_name, gradient, 40)
        freq_bands = ['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']
        xai.plot_psd_topo(model_name, gradient, freq_bands, xai_used_channel, show_channel)