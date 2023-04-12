import mne
from mne.decoding import Scaler
import numpy as np
import keras
import time
import tensorflow as tf
from model_evaluation import model_evaluation_onehot
from load_data import load_BC_data

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

import wandb
import timeit

import psutil
import GPUtil

print(psutil.cpu_percent())
print(psutil.virtual_memory().percent)

# Testing the GPUtil library for both GPU performance details
GPUtil.showUtilization()

# WANDB for tracking results
experiment_name = 'calculate_computational_cost_TFLite'
wandb.init(project="Experiments_MScThesis", entity="msc_thesis_eeg", name = experiment_name, sync_tensorboard=True)

#model_type = 'TensorFlow'
model_type = 'TensorFlow Lite'

print(model_type)

if model_type == 'TensorFlow':
    model_path = '/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/models/categorical_trained_all_TUAR/'
    model_name = 'retrained_9BC_50epochs.h5'
    model = keras.models.load_model(model_path+model_name)

elif model_type == 'TensorFlow Lite':
    # Load the TFLite model and allocate tensors.
    model_path = '/Users/bertavinas/Documents/DTU/Thesis/Code/'
    interpreter = tf.lite.Interpreter(model_path=model_path+"retrained_9BC_50epochs.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

# Timer
time0 = time.time()

use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'F9', 'F10']
test_file = ['/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/BrainCapture/20221024_User18_BC2_P0209.edf']
X1, y1, _ = load_BC_data(test_file, use_chan_bc, binary=False, e_duration=4, e_overlap=4*0.5)
X = np.concatenate(X1)
y = np.concatenate(y1)

# One hot encoding
label_encoder = LabelEncoder()
y_onehot = label_encoder.fit_transform(y)
y_test = to_categorical(y_onehot)

"""
#test_file = '/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/BrainCapture/20221024_User18_BC2_P0209.edf'
#path = '/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/labels_User18.csv'
#labels = np.genfromtxt(path, delimiter=',')

# Load 4s of EEG data
raw = mne.io.read_raw_edf(test_file, preload=True) # this takes more because its loading the whole file
# Take only 1 epoch
#raw.crop(tmax=4)

# PREPROCESSING # OJO fent el preprocessing, perq si nomes ho fem de una epoch, els filtres son differents, i la normalitzacio i tot
# 1. Pick desired channels and reorder
use_chan = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8','T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'F9', 'F10']
raw.pick_channels(use_chan)
raw.reorder_channels(use_chan)
raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
# 2. Set EEG montage
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, on_missing='ignore')
# 3. Apply notch filter
raw_notch = raw.copy().notch_filter(freqs=[50, 60]) # mod # Powerline frequency depend on the place, can be 50 or 60 Hz
# 4. Apply bandpass filter to remove dc offset and powerline noise
raw_filtered = raw_notch.copy().filter(l_freq=0.2, h_freq=45) # mod
# 5. Set EEG reference to average
ref_raw = mne.set_eeg_reference(raw_filtered, ref_channels='average', copy=True) #mod
# Create info for Scaler
info = ref_raw[0].info
# Create epochs
epochs = mne.make_fixed_length_epochs(ref_raw[0], duration=4, preload=True, overlap=4*0.5)
# Resample epochs to 256 Hz
epochs.resample(256)

# 8. Normalize data across channels
X_o = Scaler(info).fit_transform(epochs.get_data())  # Scale eeg by 1e6
X_n = Scaler(scalings='mean').fit_transform(X_o)
"""

# Timer
time1 = time.time()
print("Time load and preprocess whole file: ", time1-time0)

y_p = []
i = 0
for x in X:
    kernels, chans, samples = 1, 21, 1024
    X_i = x.reshape(1, chans, samples, kernels)  # only first epoch

    if model_type == 'TensorFlow':
        # Make prediction on test set.
        probs = model.predict(X_i)
        probs = np.array(probs)
        y_p.append(probs)

    elif model_type == 'TensorFlow Lite':
        input_data = np.array(X_i, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Make prediction on test set.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        y_p.append(output_data[0])

    # Timer
    time2 = time.time()
    print("Time to predict one epoch: ", time2-time1)

    time1 = time.time()
    i = i+1

# Timer
time3 = time.time()
print("TOTAL TIME: ", time3-time0)

print(psutil.cpu_percent())
print(psutil.virtual_memory().percent)

# Testing the GPUtil library for both GPU performance details
GPUtil.showUtilization()

# Evaluate model
classes = ['chew', 'elec', 'eyem', 'musc', 'none', 'shiv']
conf_matrix, scores = model_evaluation_onehot(np.array(y_p), y_test, classes)

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(10, 8))
sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="d", ax=ax)

ax.set_title('Confusion Matrix', weight='bold', fontsize=23)
ax.set_xlabel('\nPredicted Values', weight='bold', fontsize=18)
ax.set_ylabel('Actual Values ', weight='bold', fontsize=18)

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(classes, fontsize=18)
ax.yaxis.set_ticklabels(classes, fontsize=18)

# figure = ax.get_figure()
fig.savefig('/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/' + model_type + '_User18.pdf', bbox_inches='tight')

# The scope of these changes made to
# pandas settings are local to with statement.
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(scores)
