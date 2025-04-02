import os
import mne
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import scipy.signal

from models import *
from func import *

mne.set_log_level(verbose=0)

class Interpretation:
    def __init__(self, args):
        self.rootpath = args.savepath + '/xai/'
        self.sfreq = args.sfreq

        os.makedirs(self.rootpath, exist_ok=True)
    
    def plot_psd(self, model_name, gradient, limit=40):
        classes = ['Healthy Control', 'Schizophrenia']
        limit_freq = int(self.sfreq // 2) if limit is None else limit

        fig, ax = plt.subplots(figsize=(18, 5))

        global_y_min = float('inf')  
        global_y_max = float('-inf') 

        for idx, class_name in enumerate(classes):
            if len(gradient[idx]) == 0:
                continue

            freqs, psds = [], []
            for subject_gradient in gradient[idx]:
                f, p = scipy.signal.welch(
                    subject_gradient,
                    self.sfreq,
                    nperseg=int(self.sfreq),
                    noverlap=int(self.sfreq // 2)
                )
                freqs.append(f)
                psds.append(np.mean(np.mean(abs(p), axis=0), axis=0))  # p: (segments, channels, frequencies)

            f = [round(x) for x in freqs[0]]

            psd_norm = []
            for psd in psds:
                tmp_data = psd[:limit_freq]
                if tmp_data.max() - tmp_data.min() != 0:
                    psd_norm.append((tmp_data - tmp_data.min()) / (tmp_data.max() - tmp_data.min()))

            psd_norm = np.mean(np.array(psd_norm), axis=0)
            ax.semilogy(f[:limit_freq], psd_norm, linewidth=5, label=class_name)

            y_min, y_max = ax.get_ylim()
            global_y_min = min(global_y_min, y_min)
            global_y_max = max(global_y_max, y_max)

        ax.set_ylim([global_y_min, global_y_max])
        ax.set_xlim([0, limit_freq])
        ax.set_xticks(np.arange(0, limit_freq + 1, 5))

        ax.set_xlabel("Frequency [Hz]", labelpad=10, fontsize=26)
        ax.set_ylabel("PSD", labelpad=10, fontsize=26)
        ax.set_title("Saliency PSD", fontsize=30, pad=25, fontweight='bold')

        ax.tick_params(axis='both', labelsize=24)
        ax.yaxis.set_tick_params(labelleft=True)

        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)

        # Add frequency band boundaries and labels
        for x_tick in [4, 8, 12, 30]:
            ax.axvline(x=x_tick, color=(0.5, 0.5, 0.5), linestyle='--', linewidth=1.7)

        ax.text(0.05, 0.04, 'δ', ha='center', va='center', fontsize=22, transform=ax.transAxes)
        ax.text(0.15, 0.04, 'θ', ha='center', va='center', fontsize=22, transform=ax.transAxes)
        ax.text(0.25, 0.04, 'α', ha='center', va='center', fontsize=22, transform=ax.transAxes)
        ax.text(0.525, 0.04, 'β', ha='center', va='center', fontsize=22, transform=ax.transAxes)
        ax.text(0.875, 0.04, 'γ', ha='center', va='center', fontsize=22, transform=ax.transAxes)

        ax.legend(fontsize=14, loc='upper right')

        plt.tight_layout()
        plt.savefig(self.rootpath + model_name + '_psd.png')

    def plot_psd_topo(self, model_name, gradient, freq_bands, xai_used_channel, show_channel):
        # set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        xai_ch_pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in xai_used_channel])
        show_ch_pos = np.array([montage.get_positions()['ch_pos'][ch] for ch in show_channel])

        xai_ch_pos[:, 1] += 0.01
        show_ch_pos[:, 1] += 0.01

        classes = ['Control', 'Schizophrenia']
        freq_range = {'delta': [0, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30], 'gamma': [30, 40], 'all': [0, 40]}

        rows = 1
        cols = int(np.ceil(len(classes) / rows))

        classes_psd = {}

        # Step 1: compute PSD for each class
        for idx, class_name in enumerate(classes):
            if len(gradient[idx]) == 0:
                continue

            # for each subject, perform fft then average across segment
            freqs, psds = [], []
            for subject_gradient in gradient[idx]:
                f, p = scipy.signal.welch(subject_gradient, self.sfreq, nperseg=int(self.sfreq), noverlap=int(self.sfreq // 2)) 
                
                freqs.append(f)
                psds.append(np.mean(abs(p), axis=0))   # p: (segment, channel, frequency)
            
            # PSD normalization
            # psds: (subject, channel, frequency), norm subject's (channel, frequency)
            psd_norm = []
            for psd in psds:
                subject_psd = psd[:, :40]
                if subject_psd.max() - subject_psd.min() != 0:
                    psd_norm.append((subject_psd - subject_psd.min()) / (subject_psd.max() - subject_psd.min()))
            
            # average subject --> (channel, frequency)
            classes_psd[class_name] = np.mean(np.array(psd_norm), axis=0)
                
        # Step 2: Get PSD of each frequency band and plot the topomap
        for freq_bd in freq_bands:
            min_max_for_each_class = [float('inf'), float('-inf')]

            for idx, class_name in enumerate(classes):
                if len(gradient[idx]) == 0:
                    continue
                
                # average frequency bins
                freq_subband = np.mean(classes_psd[class_name][:, freq_range[freq_bd][0]:freq_range[freq_bd][1]], axis=1)
                
                min_max_for_each_class[1] = max(min_max_for_each_class[1], freq_subband.max())
                min_max_for_each_class[0] = min(min_max_for_each_class[0], freq_subband.min())
                

            plt.figure(figsize=(24, 9))
            ims, axs = [], []
            for idx, class_name in enumerate(classes):
                subband_psd = np.mean(classes_psd[class_name][:, freq_range[freq_bd][0]:freq_range[freq_bd][1]], axis=1)

                ax = plt.subplot(rows, cols, idx + 1)
                axs.append(ax)

                kwargs = {'pos': xai_ch_pos[:, 0:2],
                        'ch_type': 'eeg',
                        'sensors': False,
                        'axes': ax,
                        'show': False,
                        'outlines': 'head',
                        'sphere': (0.0, 0.0, 0.0, 0.12),
                        'extrapolate':'box'
                }
                cmap = 'Reds'  

                im, _ = mne.viz.plot_topomap(data=subband_psd, cmap=cmap, vlim=(min_max_for_each_class[0], min_max_for_each_class[1]), **kwargs)
                ims.append(im)

                plt.title(class_name + ' in ' + freq_bd.capitalize() + ' Band', fontsize=30, pad=25, fontweight='bold')

                # add `show_channel` electrods on topomap
                for name_idx, ch in enumerate(show_ch_pos):
                    ax.text(ch[0], ch[1], show_channel[name_idx], ha='center', va='center', fontsize=22, color='black')

            # set the shared colorbar
            cbar = plt.colorbar(ims[0], ax=axs, orientation='vertical', shrink=0.7)
            y_ticks = np.linspace(min_max_for_each_class[0], min_max_for_each_class[1], 2)
            cbar.ax.get_yaxis().set_ticks(y_ticks)
            cbar.ax.get_yaxis().set_ticklabels([min_max_for_each_class[0], min_max_for_each_class[1]])

            plt.savefig(self.rootpath + model_name + '_topo_' + freq_bd + '.png')
            plt.close()