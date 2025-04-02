import os
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from models import *
from func import *

mne.set_log_level(verbose=0)

class Pilot:
    def __init__(self, args, grid_param):
        self.rootpath = args.savepath + 'pilot/'
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path
        self.grid_param = grid_param

        self.window_length = args.window_length
        self.window_overlap = args.window_overlap

        self.hc_full_epoch_data = []
        self.sch_full_epoch_data = []

        self.total_fold = args.fold
        self.epochs = args.pilot_train_epoch

        self.sfreq = args.sfreq
        self.channel = args.channel

        self.seed = args.seed

    # load full data from `Control` and `Schizophrenia` folder
    def load_dataset(self, hc_subject_num, sch_subject_num):
        self.hc_full_epoch_data = get_dataset(self.dataset, 'Control', hc_subject_num, self.dataset_path, self.window_length, self.window_overlap)
        self.sch_full_epoch_data = get_dataset(self.dataset, 'Schizophrenia', sch_subject_num, self.dataset_path, self.window_length, self.window_overlap)

    # get train and validation sets for the specified fold
    def get_data_loader(self, now_fold, model_name, batch_size, dev):
        if self.dataset == 1:
            start_end_idx = [[0,2],[3,5],[6,8],[9,11],[12,13]]

            x_train = get_dataset1_fold(self.hc_full_epoch_data, self.sch_full_epoch_data, now_fold, self.total_fold - 2, self.total_fold, start_end_idx)
            x_valid = get_dataset1_fold(self.hc_full_epoch_data, self.sch_full_epoch_data, (now_fold + self.total_fold - 2) % self.total_fold, 1, self.total_fold, start_end_idx)
        else:
            hc_subject_in_fold = int(len(self.hc_full_epoch_data) / self.total_fold)
            sch_subject_in_fold = int(len(self.sch_full_epoch_data) / self.total_fold)  

            x_train = get_dataset2_fold(self.hc_full_epoch_data, self.sch_full_epoch_data, now_fold, self.total_fold - 2, self.total_fold, hc_subject_in_fold, sch_subject_in_fold)
            x_valid = get_dataset2_fold(self.hc_full_epoch_data, self.sch_full_epoch_data, (now_fold + self.total_fold - 2) % self.total_fold, 1, self.total_fold, hc_subject_in_fold, sch_subject_in_fold)
        
        train_loader = to_tensor(x_train, dev, batch_size, model_name, True)
        valid_loader = to_tensor(x_valid, dev, batch_size, model_name, False) 
        
        return train_loader, valid_loader

    def train(self, model_name):
        print('Pilot Train ----------------------')
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.rootpath, exist_ok=True)

        resultFile = Path(self.rootpath + 'pilot_result.txt')
        resultFile.touch(exist_ok=True)

        round_id = 0

        best_param = {'acc': 0}

        for batch in self.grid_param['training_batch']:
            for lr in self.grid_param['training_lr']:
                print("training {}: model={}, batch={}, lr={}".format(round_id, model_name, batch, lr))
                
                set_seed(self.seed)  

                for fold in range(self.total_fold):
                    train_loader, valid_loader = self.get_data_loader(fold, model_name, batch, dev)

                    model = getattr(sys.modules[__name__], model_name)
                    model = model(self.channel, int(self.sfreq*self.window_length), sfreq=self.sfreq)
                    model.to(dev)

                    loss_fn = nn.CrossEntropyLoss()
                    opt_fn = torch.optim.Adam(model.parameters(), lr=lr)

                    for ep in range(self.epochs):
                        train_loss, train_acc = train_an_epoch(model, train_loader, loss_fn, opt_fn, dev, model_name)
                    valid_loss, valid_acc, _ = evalate_an_epoch(model, valid_loader, loss_fn, dev, model_name)
                    
                    if valid_acc > best_param['acc']:
                        best_param['acc'] = valid_acc
                        best_param['training_batch'] = batch
                        best_param['training_lr'] = lr

                round_id += 1

        file = open(resultFile, 'a')
        file.write("{},{},{},\n".format(model_name, best_param['training_batch'], best_param['training_lr']))
        file.close()    