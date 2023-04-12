import mne
import numpy as np
import glob
import re

import keras
from mne.decoding import Scaler

# Load pretrained model from file
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/code_submission/models/'
model_type = 'DSCNN'
model_name = path+'DSCNN_categorical_4s_200epochs_dropout.h5'
model = keras.models.load_model(model_name)

# Load BC and FIL data
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/new_BrainCapture/'
new_bc_files = glob.glob(path + '*_volunteer*.edf')

use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
               'T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'F9', 'F10']

f = 0
for file in new_bc_files:
    raw = mne.io.read_raw_edf(file, preload=True)
    raw.pick_channels(use_chan_bc)
    # Create info for Scaler
    raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')

    info = raw.info

    lfs = 0.2
    hfs = 40.
    raw_notch = raw.copy().notch_filter(freqs=[50, 60]) # mod # Powerline frequency depend on the place, can be 50 or 60 Hz
    raw_filtered = raw_notch.copy().filter(l_freq=lfs, h_freq=hfs) # mod
    ref_raw = mne.set_eeg_reference(raw_filtered, ref_channels='average', copy=True)  # mod
    #ref_raw[0].plot(block=True, duration = 10, scalings='auto', n_channels=21)

    # make epochs
    # Create epochs
    epochs = mne.make_fixed_length_epochs(ref_raw[0], duration=4, preload=True, overlap=4*0.5)

    # Resample epochs to 256 Hz
    epochs.resample(256)
    X = epochs.get_data()

    # 8. Normalize data across channels
    X_o = Scaler(info).fit_transform(X)  # Scale eeg by 1e6
    X_n = Scaler(scalings='mean').fit_transform(X_o)

    print(X_n.shape)

    # Predict data
    probs = model.predict(X_n)  # calculate time to just predict one sample
    probs = np.array(probs)

    labels = probs.argmax(axis=-1)
    print(labels)

    art_dict = {0: 'chew', 1: 'elec', 2: 'eyem', 3: 'musc', 4: 'none', 5: 'shiv'}

    i = 0
    onset = []
    dur = []
    desc = []
    for x in X_n:
        onset.append(i*2)
        dur.append(2)
        desc.append(art_dict[labels[i]])
        print(i*2, 2, art_dict[labels[i]])
        i = i+1

    pred_annot = mne.Annotations(onset=onset,  # in seconds
                               duration=dur,  # in seconds, too
                               description=desc)

    ref_raw[0].set_annotations(pred_annot)
    ref_raw[0].plot(block=True, duration=10, scalings='auto', n_channels=21)
    print(file)
    file_name = re.findall(r'volunteer.+.edf', file)[0]
    mne.export.export_raw('/Users/bertavinas/Documents/DTU/Thesis/Code/examples_annotated_files/annotated_'+file_name, ref_raw[0], fmt='edf')
    f = f+1