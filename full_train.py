import os
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from models import *
from func import *

mne.set_log_level(verbose=0)

class Full:
    def __init__(self, args):
        self.rootpath = args.savepath
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        self.window_length = args.window_length
        self.window_overlap = args.window_overlap

        self.hc_full_epoch_data = []
        self.sch_full_epoch_data = []

        self.total_fold = args.fold
        self.epochs = args.full_train_epoch

        self.sfreq = args.sfreq
        self.channel = args.channel

        self.seed = args.seed

        self.best_param = {}

    # load the best parameters from the pilot training phase
    def get_best_param(self):
        f = open(self.rootpath + 'pilot/pilot_result.txt')
        for line in f.readlines():
            params = line.split(',')
            self.best_param[params[0]] = {'training_batch': int(params[1]),
                                            'training_lr': float(params[2])}
        f.close()
    
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
        print('Full Train ----------------------')
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        savepath = self.rootpath + 'full/' + model_name + '/'
        os.makedirs(savepath, exist_ok=True)  
        
        set_seed(self.seed)  

        for fold in range(self.total_fold):
            train_loader, valid_loader = self.get_data_loader(fold, model_name, self.best_param[model_name]['training_batch'], dev)

            model = getattr(sys.modules[__name__], model_name)
            model = model(self.channel, int(self.sfreq*self.window_length), sfreq=self.sfreq)
            model.to(dev)

            loss_fn = nn.CrossEntropyLoss()
            opt_fn = torch.optim.Adam(model.parameters(), lr=self.best_param[model_name]['training_lr'])

            best_valid_model = {'acc': 0, 'model': None}
            for ep in range(self.epochs):
                train_loss, train_acc = train_an_epoch(model, train_loader, loss_fn, opt_fn, dev, model_name)
                valid_loss, valid_acc, _ = evalate_an_epoch(model, valid_loader, loss_fn, dev, model_name)

                if ep >= 30 and valid_acc >= best_valid_model['acc']:
                    best_valid_model['acc'] = valid_acc
                    best_valid_model['model'] = model.state_dict()
            
            torch.save(best_valid_model['model'], savepath + 'fold' + str(fold) + '_best_model.pth')


            
    
