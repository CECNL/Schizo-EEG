import argparse
import time
import os
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from captum.attr import Saliency
from matplotlib.lines import Line2D

from IFNet import IFNet

# fix the seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# get dataset 1 train/valid/test data
def get_dataset1_fold(da, db, now_fold, include_fold, total_fold, start_end_idx):
    result = []
    
    for i in range(include_fold):
        f = (now_fold + i) % total_fold
        result += da[start_end_idx[f][0] : start_end_idx[f][1]+1]

    for i in range(include_fold):
        f = (now_fold + i) % total_fold
        result += db[start_end_idx[f][0] : start_end_idx[f][1]+1]
    
    return result

# get dataset 2 (KMUH) train/valid/test data
def get_dataset2_fold(da, db, now_fold, include_fold, total_fold, hc_subject_in_fold, sch_subject_in_fold):
    result = []         
    for i in range(include_fold):  
        result += da[((now_fold + i) % total_fold) * hc_subject_in_fold : ((now_fold + i) % total_fold) * hc_subject_in_fold + hc_subject_in_fold]

    for i in range(include_fold):
        result += db[((now_fold + i) % total_fold) * sch_subject_in_fold : ((now_fold + i) % total_fold) * sch_subject_in_fold + sch_subject_in_fold]
    return result

def to_tensor(raw, dev, batch_size, model_name, shuffle=False):
    x = []
    y = []
    for r in raw:
        x.append(r[-2])
        y.append(r[-1])
    x = np.concatenate(x)
    y = np.concatenate(y)

    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y).type(torch.LongTensor)
    tensor_y = F.one_hot(tensor_y, 2)

    # if model_name != 'Oh_CNN' and model_name != 'SzHNN':
    #     tensor_x = tensor_x.unsqueeze(1)

    tensor_x = tensor_x.to(dev)
    tensor_y = tensor_y.to(dev)

    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

def train_an_epoch(model, data_loader, loss_fn, optimizer, device, model_name):
    model.train()

    a, b = 0, 0  # hit sample, total sample
    epoch_loss = np.zeros((len(data_loader), ))
    for i, (x_batch, y_batch) in enumerate(data_loader):
        x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)
        optimizer.zero_grad()

        if model_name == 'MBSzEEGNet':
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
        elif model_name == 'MBCFCNet':
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
        else:
            output = model(x_batch)
            loss = loss_fn(output, y_batch.argmax(dim=1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss[i] = loss.item()
        b += y_batch.size(0)
        a += torch.sum(y_batch.argmax(dim=1) == output.argmax(dim=1)).item()
    return epoch_loss.mean(), a / b  # return the loss and acc


def evalate_an_epoch(model, data_loader, loss_fn, device, model_name):
    predict_label = []

    model.eval()
    with torch.no_grad():
        a, b = 0, 0 # hit sample, total sample
        epoch_loss = np.zeros((len(data_loader), ))
        for i, (x_batch, y_batch) in enumerate(data_loader):
            x_batch.requires_grad = False
            x_batch, y_batch = x_batch.to(device, dtype=torch.float), y_batch.to(device, dtype=torch.float)

            if model_name == 'MBSzEEGNet':
                output = model(x_batch)
                loss = loss_fn(output, y_batch)
            else:
                output = model(x_batch)
                loss = loss_fn(output, y_batch.argmax(dim=1))

            epoch_loss[i] = loss.item()
            b += y_batch.size(0)
            a += torch.sum(y_batch.argmax(dim=1) == output.argmax(dim=1)).item()

            predict_label.extend(output.argmax(dim=1).tolist()) # get predict labels and return
    return epoch_loss.mean(), a / b, predict_label # return the loss and acc

from scipy import signal
import numpy as np
import mne

def bandpass_filter(data, sfreq, band):
    b, a = signal.butter(N=5, Wn=band, btype='bandpass', fs=sfreq)
    return signal.lfilter(b, a, data, axis=-1)

def get_dataset(dataset_id, class_name, subject_num, dataset_path, window_length, window_overlap, sfreq=125, filter_bank=[(4, 16), (16, 40)]):
    full_epoch_data = []
    class_id = 0 if class_name == 'Control' else 1
    file_name = 'h' if class_name == 'Control' else 's'

    for i in range(1, subject_num + 1):
        subject_id = str(i).zfill(2)

        if dataset_id == 1:
            raw_data = mne.io.read_raw_eeglab(dataset_path + class_name + '/' + file_name + subject_id + '.set', preload=True)
            epochs = mne.make_fixed_length_epochs(raw_data, duration=window_length, overlap=window_overlap)
            epochs_data = epochs.get_data()  # shape: (N, C, T)
        else:
            epochs_data = np.load(dataset_path + class_name + '/' + file_name + subject_id + '.npy')  # shape: (N, C, T)

        # z-score normalization
        mean = epochs_data.mean()
        std = epochs_data.std()
        x = (epochs_data - mean) / std  # shape: (N, C, T)

        # apply filter bank: (N, C, T) → list of (N, C, T)
        filtered_list = [bandpass_filter(x, sfreq, band) for band in filter_bank]

        # concatenate in channel dimension: (N, C*len(filter_bank), T)
        x_concat = np.concatenate(filtered_list, axis=1)

        y = [class_id] * x_concat.shape[0]
        full_epoch_data.append([file_name + subject_id, x_concat, y])

    return full_epoch_data


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

                    model = IFNet(self.channel, int(self.sfreq*self.window_length), sfreq=self.sfreq)
                    model.to(dev)

                    loss_fn = nn.CrossEntropyLoss()
                    opt_fn = torch.optim.Adam(model.parameters(), lr=lr)

                    import time
                    # start = time.time()
                    for ep in range(self.epochs):
                        start = time.time()
                        train_loss, train_acc = train_an_epoch(model, train_loader, loss_fn, opt_fn, dev, model_name)
                        end = time.time()
                        # print(f"Training time per sample: {end - start:.6f} seconds")

                    start = time.time()
                    valid_loss, valid_acc, _ = evalate_an_epoch(model, valid_loader, loss_fn, dev, model_name)
                    end = time.time()
                    # print(f"Inference time per sample: {end - start:.6f} seconds")
                    
                    if valid_acc > best_param['acc']:
                        best_param['acc'] = valid_acc
                        best_param['training_batch'] = batch
                        best_param['training_lr'] = lr

                round_id += 1

        file = open(resultFile, 'a')
        file.write("{},{},{},\n".format(model_name, best_param['training_batch'], best_param['training_lr']))
        file.close()    


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

            # model = getattr(sys.modules[__name__], model_name)
            model = IFNet(self.channel, int(self.sfreq*self.window_length), sfreq=self.sfreq)
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

mne.set_log_level(verbose=0)

class Test:
    def __init__(self, args):
        self.rootpath = args.savepath
        self.dataset = args.dataset
        self.dataset_path = args.dataset_path

        self.total_fold = args.fold

        self.channel = args.channel
        self.sfreq = args.sfreq
        self.window_length = args.window_length
        self.window_overlap = args.window_overlap 

        self.best_param = {}
        self.saliency_maps = {}

        #### only for dataset 1
        self.test_subject_amount_ineach_fold = [3, 3, 3, 3, 2]
        ####

        self.seed = args.seed

    def get_best_param(self):
        f = open(self.rootpath + 'pilot/pilot_result.txt')
        for line in f.readlines():
            params = line.split(',')
            self.best_param[params[0]] = {'training_batch': int(params[1]),
                                            'training_lr': float(params[2])}
        f.close()

    def load_dataset(self, hc_subject_num, sch_subject_num):
        self.hc_full_epoch_data = get_dataset(self.dataset, 'Control', hc_subject_num, self.dataset_path, self.window_length, self.window_overlap)
        self.sch_full_epoch_data = get_dataset(self.dataset, 'Schizophrenia', sch_subject_num, self.dataset_path, self.window_length, self.window_overlap)

    def evalate_and_get_gradient(self, model, data_loader, dev, model_name, class_id):
        predict_label = []
        label_list = []
        gradient_list = []
        
        saliency_inst = Saliency(model)
        if model_name != 'SzHNN':
            model.eval()
        else:
            model.train()

        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(data_loader):
                x_batch.requires_grad = False

                x_batch, y_batch = x_batch.to(dev, dtype=torch.float), y_batch.to(dev, dtype=torch.float)

                if model_name == 'MBCFCNet':
                    output = model(x_batch)
                else:
                    output = model(x_batch)
                
                label_list.append(y_batch.argmax(dim=1).detach().cpu().numpy())
                predict_label.extend(output.argmax(dim=1).tolist()) 
                
                
                x_batch.requires_grad = True
                gradient_list.append(
                    saliency_inst.attribute(x_batch, target=y_batch.argmax(dim=1).detach().cpu().numpy().tolist(), abs=False,).detach().cpu().numpy()
                )
            
            label_list = np.concatenate(label_list)
            gradient_list = np.concatenate(gradient_list)
            if gradient_list.shape[1] == 1:
                gradient_list = np.squeeze(gradient_list, axis=1)

            arr = np.array([index for index, (x, y) in enumerate(zip(label_list, predict_label)) if x == y and x == class_id])
            if arr.shape[0] != 0:
                gradient = gradient_list[arr]
            else:
                gradient = np.array([])

        return predict_label, gradient

    def get_subject_test_loader(self, data_path, subject, model_name, dev, sfreq=125, filter_bank=[(4, 16), (16, 40)]):
        class_name = data_path.split('/')[-1][0]     # health control: h; schizophrenia: s
        class_id = 0 if class_name == "h" else 1     # health control: 0; schizophrenia: 1

        full_epoch_data = []

        if self.dataset == 1:
            raw_data = mne.io.read_raw_eeglab(data_path, preload=True)
            epochs = mne.make_fixed_length_epochs(raw_data, duration=self.window_length, overlap=self.window_overlap)
            epochs_data = epochs.get_data()
        else:
            epochs_data = np.load(data_path)

        # z-score norm. to each subject
        mean = epochs_data.mean()
        std = epochs_data.std()
        x = (epochs_data - mean) / std

        # apply filter bank: (N, C, T) → list of (N, C, T)
        filtered_list = [bandpass_filter(x, sfreq, band) for band in filter_bank]

        # concatenate in channel dimension: (N, C*len(filter_bank), T)
        x_concat = np.concatenate(filtered_list, axis=1)
        y = [class_id] * x_concat.shape[0]
        # full_epoch_data.append([file_name + subject_id, x_concat, y])
        # y = [class_id] * epochs_data.shape[0]

        full_epoch_data.append([class_name + str(subject), x_concat, y])

        test_loader = to_tensor(full_epoch_data, dev, self.best_param[model_name]['training_batch'], model_name, False)

        return test_loader
    
    def get_fold_test_loader(self, now_fold, model_name, batch_size, dev):
        if self.dataset == 1:
            start_end_idx = [[0,2],[3,5],[6,8],[9,11],[12,13]]
            
            x_test = get_dataset1_fold(self.hc_full_epoch_data, self.sch_full_epoch_data, (now_fold + self.total_fold - 2 + 1) % self.total_fold, 1, self.total_fold, start_end_idx)
        else:
            hc_subject_in_fold = int(len(self.hc_full_epoch_data) / self.total_fold)
            sch_subject_in_fold = int(len(self.sch_full_epoch_data) / self.total_fold)  

            x_test = get_dataset2_fold(self.hc_full_epoch_data, self.sch_full_epoch_data, (now_fold + self.total_fold - 2 + 1) % self.total_fold, 1, self.total_fold, hc_subject_in_fold, sch_subject_in_fold)
        
        test_loader = to_tensor(x_test, dev, batch_size, model_name, False)
        
        return test_loader

    def get_predict_result(self, data_path, test_loader, dev, model_name, best_model):
        class_name = data_path.split('/')[-1][0]     # health control: h; schizophrenia: s
        class_id = 0 if class_name == "h" else 1     # health control: 0; schizophrenia: 1
        
        predict_label, gradient = self.evalate_and_get_gradient(best_model, test_loader, dev, model_name, class_id)

        correct_count = predict_label.count(class_id)

        predict_class = max(predict_label, key=predict_label.count)

        return correct_count, len(predict_label), predict_class, gradient

    def test(self, model_name, hc_subject_num, sch_subject_num):
        print('Testing ----------------------')
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        file = Path(self.rootpath + '/full/test_result.txt')
        file.touch(exist_ok=True)

        self.get_best_param()

        result_txt = open(file, mode="a")
        result_txt.write("model, seed, batch size, learning rate, Accuracy (Seg.), Accuracy (Sub.), Sensitivity, Specificity, Precision, F1_score \n")
        result_txt.close()

        set_seed(self.seed)
        model_path = self.rootpath + 'full/' + model_name

        # segment-wise
        correct_segment, total_segment = 0, 0

        # subject-wise
        tp, fp, tn, fn = 0, 0, 0, 0

        # gradient (saliency)
        saliency_maps = {0: [], 1: []}

        hc_idx, sch_idx = 1, 1
        for fold in range(1, self.total_fold + 1):
            # load fold model
            # best_model = getattr(sys.modules[__name__], model_name)
            best_model = IFNet(self.channel, int(self.sfreq*self.window_length), sfreq=self.sfreq)
            best_model.to(dev)

            best_model.load_state_dict(torch.load(model_path + '/fold' + str(fold % self.total_fold) + '_best_model.pth'))
            best_model.eval()

            num = self.test_subject_amount_ineach_fold[fold - 1] if self.dataset == 1 else hc_subject_num // self.total_fold
            for h in range(num):
                if self.dataset == 1:
                    data_path = self.dataset_path + 'Control/h' + str(hc_idx).zfill(2) + '.set'
                else:
                    data_path = self.dataset_path + 'Control/h' + str(hc_idx).zfill(2) + '.npy'
                test_loader = self.get_subject_test_loader(data_path, hc_idx, model_name, dev)

                correct_count, segment_count, predict_class, gradient = self.get_predict_result(data_path, test_loader, dev, model_name, best_model)
                
                if gradient.shape[0] != 0:
                    saliency_maps[0].append(gradient)

                correct_segment += correct_count
                total_segment += segment_count

                if predict_class == 0:
                    tn += 1 # hc --> hc
                else:
                    fp += 1 # hc --> sz
                
                hc_idx += 1

            num = self.test_subject_amount_ineach_fold[fold - 1] if self.dataset == 1 else sch_subject_num // self.total_fold
            for s in range(num):
                if self.dataset == 1:
                    data_path = self.dataset_path + 'Schizophrenia/s' + str(sch_idx).zfill(2) + '.set'
                else:
                    data_path = self.dataset_path + 'Schizophrenia/s' + str(sch_idx).zfill(2) + '.npy'
                test_loader = self.get_subject_test_loader(data_path, sch_idx, model_name, dev)

                correct_count, segment_count, predict_class, gradient = self.get_predict_result(data_path, test_loader, dev, model_name, best_model)

                if gradient.shape[0] != 0:
                    saliency_maps[1].append(gradient)

                correct_segment += correct_count
                total_segment += segment_count

                if predict_class == 1:
                    tp += 1 # sz --> sz
                else:
                    fn += 1 # sz --> hc
                
                sch_idx += 1
        
        segment_accuracy = correct_segment / total_segment
        subject_accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        precision = tp / (tp + fp)
        f1_score = (2 * precision * sensitivity) / (precision + sensitivity)

        result_txt = open(file, mode="a")
        result_txt.write("{}, {}, {}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f} \n".format(model_name, self.best_param[model_name]['training_batch'], self.best_param[model_name]['training_lr'], segment_accuracy*100, subject_accuracy*100, sensitivity*100, specificity*100, precision*100, f1_score*100))
        result_txt.close()

        self.saliency_maps = saliency_maps
    
        return self.saliency_maps
            


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=1, help='dataset you want to use', choices=[1, 2])
    parser.add_argument('--dataset_path', type=str, default='./dataset_1/', help='the path contains the final preprocessed folders: Control and Schizophrenia')
    parser.add_argument('--savepath', type=str, default='./IFNet_test1/', help='the folder path for saving the training and testing result')

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

    model_list = ['IFNet'] if args.model == 'All' else [args.model]    

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
        
        # #### Test
        test = Test(args)
        gradient = test.test(model_name, hc_subject_num, sch_subject_num)  # gradient = {0: [subject, epochs, channel, timepoints], 1: [subject, epochs, channel, timepoints]}