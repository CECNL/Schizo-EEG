import random
import numpy as np
import mne
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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

    if model_name != 'Oh_CNN' and model_name != 'SzHNN':
        tensor_x = tensor_x.unsqueeze(1)

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

def get_dataset(dataset_id, class_name, subject_num, dataset_path, window_length, window_overlap):
    full_epoch_data = []

    class_id = 0 if class_name == 'Control' else 1
    file_name = 'h' if class_name == 'Control' else 's'

    for i in range(1, subject_num+1):
        subject_id = str(i).zfill(2)

        if dataset_id == 1:
            raw_data = mne.io.read_raw_eeglab(dataset_path + class_name + '/' + file_name + subject_id + '.set', preload=True)
            epochs = mne.make_fixed_length_epochs(raw_data, duration=window_length, overlap=window_overlap)
            epochs_data = epochs.get_data()
        else:
            epochs_data = np.load(dataset_path + class_name + '/' + file_name + subject_id + '.npy')

        # z-score norm. to each subject (after cut)
        mean = epochs_data.mean()
        std = epochs_data.std()
        x = (epochs_data - mean) / std

        y = [class_id] * epochs_data.shape[0]

        full_epoch_data.append([file_name+subject_id, x, y])

    return full_epoch_data

