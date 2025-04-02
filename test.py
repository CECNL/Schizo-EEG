import os
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from captum.attr import Saliency
from matplotlib.lines import Line2D

from models import *
from func import *

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

    def get_subject_test_loader(self, data_path, subject, model_name, dev):
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

        y = [class_id] * epochs_data.shape[0]

        full_epoch_data.append([class_name + str(subject), x, y])

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
            best_model = getattr(sys.modules[__name__], model_name)
            best_model = best_model(self.channel, int(self.sfreq*self.window_length), sfreq=self.sfreq)
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

    def tsne(self, model_name):
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import MinMaxScaler
        print('t-sne...')
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.get_best_param()

        savepath = self.rootpath + 'tsne/'
        os.makedirs(savepath, exist_ok=True)  

        model_path = self.rootpath + 'full/' + model_name
        

        for fold in range(self.total_fold):
            test_loader = self.get_fold_test_loader(fold, model_name, self.best_param[model_name]['training_batch'], dev)

            model = getattr(sys.modules[__name__], model_name)
            model = model(self.channel, int(self.sfreq*self.window_length), sfreq=self.sfreq, tsne=True)
            model.to(dev)
            model.load_state_dict(torch.load(model_path + '/fold' + str(fold % self.total_fold) + '_best_model.pth'))

            loss_fn = nn.CrossEntropyLoss()
            opt_fn = torch.optim.Adam(model.parameters(), lr=self.best_param[model_name]['training_lr'])

            if model_name != 'SzHNN':
                model.eval()
            else:
                model.train()

            tsne_info = []
            labels = []
            with torch.no_grad():
                for i, (x_batch, y_batch) in enumerate(test_loader):
                    x_batch.requires_grad = False

                    x_batch, y_batch = x_batch.to(dev, dtype=torch.float), y_batch.to(dev, dtype=torch.float)

                    features = model(x_batch)
                    
                    tsne_info.extend(features.cpu())
                    labels.extend(np.array(y_batch.argmax(dim=1).cpu()))

            tsne_info = np.array(tsne_info)
            labels = np.array(labels).reshape(-1, 1)
            
            tsne = TSNE(n_components=2, random_state=42, verbose=1, n_iter=3000)
            features_2d = tsne.fit_transform(tsne_info)

            scaler = MinMaxScaler(feature_range=(0, 1))
            features_norm = scaler.fit_transform(features_2d)

            plt.figure(figsize=(10, 8))
            colors = ['blue' if label == 0 else 'red' for label in labels]
            plt.scatter(
                [point[0] for point in features_norm],  # X axis
                [point[1] for point in features_norm],  # Y axis
                c=colors,                             # point color
                alpha=0.5,                            # point transparency
            )
            
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Healthy Control'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Schizophrenia')
            ]

            plt.legend(handles=legend_elements, fontsize=14, loc='upper right')  # set legend
        
            plt.savefig(savepath + model_name + '_f' + str(fold) + '_tsne.png')
            plt.close()
            

