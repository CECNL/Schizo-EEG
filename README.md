# Schizo-EEG
Repetition code of the model for the paper "Enhancing EEG-Based Schizophrenia Diagnosis with Explainable Multi-Branch Deep Learning" in PyTorch

## Environment Setup
### Step 1: Create Python Environment
```
conda create --name Schizo_EEG python=3.9.7
conda activate Schizo_EEG
```
### Step 2: Install Requirements
```
pip3 install -r requirements.txt
```
<!--
```
numpy==1.26.2
scikit-learn==1.0.2
matplotlib==3.8.2
torch==2.1.1
captum==0.7.0
scipy==1.11.4
EDFlib-Python==1.0.8
```
-->
### Step 3: Prepare the Dataset <br>
The IPIN-SZ-EEG dataset used in our paper is available [here](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441) <br>
Put the data into a folder called `dataset 1`

## Run the Code
> python main.py


## Optional Arguments
<table style="width:75%; margin: 0 auto">
        <tr>
            <td style="width: 20%;"><strong>Parameter</strong></td>
            <td style="width: 15%;"><strong>Default</strong></td>
            <td style="width: 65%;"><strong>Description</strong></td>
        </tr>
        <tr>
            <td>--dataset</td>
            <td>1</td>
            <td>dataset you want to use (choices: 1, 2) <br>Dataset 1 for IPIN-SZ-EEG dataset; Dataset 2 for KMUH-SZ-EEG dataset</td>
        </tr>
        <tr>
            <td>--dataset_path</td>
            <td>'./dataset 1/'</td>
            <td>the path contains the final preprocessed folders: Control and Schizophrenia</td>
        </tr>
        <tr>
            <td>--savepath</td>
            <td>'./checkpoints/'</td>
            <td>the folder path for saving the training and testing result</td>
        </tr>
        <tr>
            <td>--pilot_train_epoch</td>
            <td>30</td>
            <td>number of epochs that you want to train in pilot train stage</td>
        </tr>
        <tr>
            <td>--full_train_epoch</td>
            <td>100</td>
            <td>number of epochs that you want to train in full train stage</td>
        </tr>
        <tr>
            <td>--fold</td>
            <td>5</td>
            <td>split data into ? folds</td>
        </tr>
        <tr>
            <td>--model</td>
            <td>All</td>
            <td>choose EEG decoding model <br>(choices: 'All', 'Oh_CNN', 'SzHNN', 'EEGNet', 'SCCNet', 'ShallowConvNet', 'MBSzEEGNet')</td>
        </tr>
        <tr>
            <td>--sfreq</td>
            <td>125</td>
            <td>sampling rate of data</td>
        </tr>
        <tr>
            <td>--channel</td>
            <td>None</td>
            <td>channel of data <br>(if None, channels default to 19 for dataset1 and 20 for dataset2; otherwise, the specified integer is used)</td>
        </tr>
        <tr>
            <td>--window_length</td>
            <td>5.0</td>
            <td>sliding window length (second) of each segment when splitting the data</td>
        </tr>
        <tr>
            <td>--window_overlap</td>
            <td>4.0</td>
            <td>the overlap duration between each segment and the previous one</td>
        </tr>
    </table>
