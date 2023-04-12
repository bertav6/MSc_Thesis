"""
    Data preprocessing functions
"""
import numpy as np
import mne
import pandas as pd

from mne.decoding import Scaler
import matplotlib.pyplot as plt

def data_preprocessing(raw, file_name, use_chan, dataset, binary, e_duration, e_overlap):
    """
        Preprocessing pipeline for raw EEG

        :return:
        X_n -> Matrix with EGG data in epochs
        y -> Labels vector
    """

    use_chan_tuar = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
                'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
                #'EEG T1-REF', 'EEG T2-REF']
    use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                   'T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'F9', 'F10']
    use_chan_fil = ['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF','EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T7-REF', 'EEG T8-REF', 'EEG P7-REF',
                    'EEG P8-REF', 'EEG T9-REF', 'EEG T10-REF', 'EEG Fz-REF', 'EEG Cz-REF', 'EEG Pz-REF']# , 'EEG F9-REF', 'EEG F10-REF']#, 'EEG P9-REF', 'ECG EKG-REF', 'EEG P10-REF', 'EEG TP7-REF', 'EEG TP8-REF', 'Photic-REF', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']

    new_chan_dict = dict(zip(use_chan_tuar, use_chan_bc))
    new_chan_dict_fil = dict(zip(use_chan_fil, use_chan_bc))

    f_s = 256
    lfs = 0.2
    hfs = 40.

    #fig, ax = plt.subplots(3, figsize=(10, 6))

    # 1. Pick desired channels
    raw.pick_channels(use_chan)
    # Reorder channels
    raw.reorder_channels(use_chan)

    # 2. Set montage
    if dataset == 'TUAR':
        raw.rename_channels(new_chan_dict)
    elif dataset == 'FIL':
        raw.rename_channels(new_chan_dict_fil)
    raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    #raw.plot_psd()

    # 3. Apply notch filter
    raw_notch = raw.copy().notch_filter(freqs=[50])#, 60]) # mod # Powerline frequency depend on the place, can be 50 or 60 Hz
    #raw_notch.plot_psd()

    # 4. Apply bandpass filter to remove dc offset and powerline noise
    raw_filtered = raw_notch.copy().filter(l_freq=lfs, h_freq=hfs) # mod
    #raw_filtered.plot()
    #raw_filtered.plot_psd()

    # 5. Set EEG reference to average
    ref_raw = mne.set_eeg_reference(raw_filtered, ref_channels='average', copy=True) #mod
    #ref_raw[0].plot()
    #ref_raw[0].plot_psd()

    # 6. Set annotations
    new_raw = set_annotations(ref_raw[0], file_name, dataset)
    #new_raw = set_annotations(raw_filtered, file_name, dataset) # skip set EEG to average reference

    # Create info for Scaler
    info = new_raw.info

    # 7. Get data
    if binary:
        if dataset == 'TUAR':
            X, y = filtered_epoch_binary_TUAR_annotated_data(new_raw, e_duration=e_duration, e_overlap=e_overlap)
        elif dataset == 'BC' or 'FIL':
            X, y = filtered_epoch_binary_bc_annotated_data(new_raw, e_duration=e_duration, e_overlap=e_overlap)
    else:
        if dataset == 'TUAR':
            #X, y = epoch_cat_annotated_TUAR_data(new_raw, e_duration=e_duration, e_overlap=e_overlap)
            X, y = filtered_epoch_cat_annotated_TUAR_data(new_raw, e_duration=e_duration, e_overlap=e_overlap)
        elif dataset == 'BC' or 'FIL':
            #X, y = epoch_cat_annotated_bc_data(new_raw, e_duration=e_duration, e_overlap=e_overlap)
            X, y = filtered_epoch_cat_annotated_bc_data(new_raw, e_duration=e_duration, e_overlap=e_overlap)

    # 8. Normalize data across channels
    X_o = Scaler(info).fit_transform(X)  # Scale eeg by 1e6
    X_n = Scaler(scalings='mean').fit_transform(X_o)
    #X_n = X

    return X_n, y

def filtered_epoch_binary_TUAR_annotated_data(new_bc_data, e_duration, e_overlap): #, figure, axes):
    """
    Epoch and filter binary TUAR EEG data

    :return:
    X -> matrix data in epochs (#epochs, #channels, #samples/epoch)
    y -> vector labels (#epochs)
    """

    # Create epochs
    epochs = mne.make_fixed_length_epochs(new_bc_data, duration=e_duration, preload=True, overlap=e_overlap)

    # Resample epochs to 256 Hz
    epochs.resample(256)

    # Create labels
    labels = []
    # Index of epochs to drop
    e_remove = []
    for annotation in epochs.get_annotations_per_epoch():
        if len(annotation) != 0:
            label = annotation[0][2]

            if '_' in label:  # more than one artifact e.g. chew_elec
                # label = tuple(label.split("_"))
                e_remove.append(True)  # exclude epoch
            else:
                onset = annotation[0][0]  # onset where t=0 is start of epoch
                duration = annotation[0][1]

                # if artifact starts within the epoch
                if onset >= 0:
                    a_dur = e_duration - onset
                # if artifact is during the epoch or end of artifact
                elif onset < 0:
                    a_dur = duration - abs(onset)

                if a_dur > e_duration/2:
                    label = 1
                    labels.append(label)
                    e_remove.append(False)  # include epoch
                else:
                    e_remove.append(True)  # exclude epoch
        else:
            label = 0
            labels.append(label)
            e_remove.append(False)  # include epoch

    # Drop epochs where artifact is less than 50% of the duration
    epochs.drop(e_remove)

    X = epochs.get_data()
    y = np.array(labels)

    return X, y

def filtered_epoch_binary_bc_annotated_data(new_bc_data, e_duration, e_overlap): #, figure, axes):
    """
    Epoch and filter binary BC data

    :return:
    X -> matrix data in epochs (#epochs, #channels, #samples/epoch)
    y -> vector labels (#epochs)
    """

    # Create epochs
    epochs = mne.make_fixed_length_epochs(new_bc_data, duration=e_duration, preload=True, overlap=e_overlap)

    # Resample epochs to 256 Hz
    epochs.resample(256)

    # Create labels
    labels = []
    # Index of epochs to drop
    e_remove = []
    for annotation in epochs.get_annotations_per_epoch():
        if len(annotation) != 0:
            bc_label = annotation[0][2]
            if bc_label not in ['Eyes closed', 'Look straight forward keep e', 'Mental arithmetic']: # Data is artifact
                tag = 1 # artifact
            else:
                tag = 0 # non artifact

            onset = annotation[0][0]  # onset where t=0 is start of epoch
            duration = annotation[0][1]

            # if artifact starts within the epoch
            if onset >= 0:
                a_dur = e_duration - onset
            # if artifact is during the epoch or end of artifact
            elif onset < 0:
                a_dur = duration - abs(onset)

            if a_dur > e_duration/2:
                labels.append(tag)
                e_remove.append(False)  # include epoch
            else:
                e_remove.append(True)  # exclude epoch

        else: # Don't include epochs in between exercices
            e_remove.append(True)  # exclude epoch

    # Drop epochs where artifact is less than 50% of the duration
    epochs.drop(e_remove)

    X = epochs.get_data()
    y = np.array(labels)

    return X, y

def filtered_epoch_cat_annotated_TUAR_data(new_bc_data, e_duration, e_overlap):
    """
    Function epoch newly annotated BC data

    NOTE: it only works for the data labeled as in file '20220927_User11_BC2_P0188.edf' !!!

    :return:
    X -> matrix data in epochs (#epochs, #channels, #samples/epoch)
    y -> vector labels (#epochs)
    """

    # Create epochs
    epochs = mne.make_fixed_length_epochs(new_bc_data, duration=e_duration, preload=True, overlap=e_overlap)

    # Resample epochs to 256 Hz
    epochs.resample(256)

    # Mean and Std for subject
    #data = epochs.get_data()
    #mean = np.mean(data)
    #std = np.std(data)

    # Create labels
    labels = []
    # Index of epochs to drop
    e_remove = []
    for annotation in epochs.get_annotations_per_epoch():
        if len(annotation) != 0:
            label = annotation[0][2]

            if '_' in label: # more than one artifact e.g. chew_elec
                #label = tuple(label.split("_"))
                e_remove.append(True)  # exclude epoch
            else:
                onset = annotation[0][0]  # onset where t=0 is start of epoch
                duration = annotation[0][1]

                # if artifact starts within the epoch
                if onset >= 0:
                    a_dur = e_duration - onset
                # if artifact is during the epoch or end of artifact
                elif onset < 0:
                    a_dur = duration - abs(onset)

                if a_dur > e_duration/2:
                    labels.append(label)
                    e_remove.append(False)  # include epoch
                else:
                    e_remove.append(True)  # exclude epoch
        else:
            label = 'none'  # label for non artifacts
            labels.append(label)
            e_remove.append(False)  # include epoch

    epochs.drop(e_remove)

    X = epochs.get_data()
    #X_n = (X - mean) / std
    y = labels

    return X, y

def filtered_epoch_cat_annotated_bc_data(new_bc_data, e_duration, e_overlap):
    """
    Function epoch newly annotated BC data

    NOTE: it only works for the data labeled as in file '20220927_User11_BC2_P0188.edf' !!!

    :return:
    X -> matrix data in epochs (#epochs, #channels, #samples/epoch)
    y -> vector labels (#epochs)
    """
    # Tranform labels form TUAR (training) to BC (test)
    trans_labels = {'Chewing': 'chew', 'Electrode pops': 'elec', 'Eye blinking': 'eyem', 'Eyes closed': 'none', 'Jaw clenching': 'chew',
     'Lateral eye movement': 'eyem', 'Lateral head movement': 'musc', 'Look straight forward keep e': 'none', 'Mental arithmetic': 'none',
     'Shivering': 'shiv', 'none': 'none'}

    # Create epochs
    epochs = mne.make_fixed_length_epochs(new_bc_data, duration=e_duration, preload=True, overlap=e_overlap)

    # Resample epochs to 256 Hz
    epochs.resample(256)

    # Mean and Std for subject
    #data = epochs.get_data()
    #mean = np.mean(data)
    #std = np.std(data)

    # Create labels
    labels = []
    # Index of epochs to drop
    e_remove = []
    for annotation in epochs.get_annotations_per_epoch():
        if len(annotation) != 0:
            bc_label = annotation[0][2]
            if bc_label not in ['Eyes closed', 'Look straight forward keep e', 'Mental arithmetic']: # Data is artifact
                tag = trans_labels[bc_label] # artifact
            else:
                tag = 'none' # non artifact

            onset = annotation[0][0]  # onset where t=0 is start of epoch
            duration = annotation[0][1]

            # if artifact starts within the epoch
            if onset >= 0:
                a_dur = e_duration - onset
            # if artifact is during the epoch or end of artifact
            elif onset < 0:
                a_dur = duration - abs(onset)

            if a_dur > e_duration/2:
                labels.append((tag,))
                e_remove.append(False)  # include epoch
            else:
                e_remove.append(True)  # exclude epoch

        else: # Don't include epochs in between exercices
            e_remove.append(True)  # Exclude epoch

    epochs.drop(e_remove)
    X = epochs.get_data()
    #X_n = (X - mean)/std
    y = labels

    return X, y

def set_annotations(raw, file_name, dataset):
    """
    Set annotate for the EEG data.

    :return:
    new_data -> newly annotated EEG data
    """
    if dataset == 'TUAR':
        df_labels = pd.read_csv('/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/TUAR/csv/labels_01_tcp_ar.csv', header=4,
                                skiprows=[6], skipinitialspace=True)
        df_labels.drop(columns = ['channel_label'], axis=1, inplace=True)
        df_labels.drop_duplicates(inplace=True)

        f_name = file_name.split('/')
        df_recording = df_labels[df_labels['# key'] == f_name[-1].replace(".edf", "")]
        description = df_recording['artifact_label']
        start_time = df_recording['start_time']
        end_time = df_recording['stop_time']
        duration = end_time - start_time

        # Set new annotations
        new_annot = mne.Annotations(onset=start_time, duration=duration, description=description)
        new_data = raw.copy().set_annotations(new_annot)
    elif dataset == 'BC':
        new_data = new_annotated_bc_data(raw)
    elif dataset == 'FIL':
        new_data = new_annotated_bc_data(raw)

    return new_data

def new_annotated_bc_data(bc_data):
    """
    Function annotate exercices in BC data

    NOTE: it only works for the data labeled as in file '20220927_User11_BC2_P0188.edf' !!!

    :return:
    new_bc_data -> newly annotated BC data
    """
    # Create event_id
    events, event_id = mne.events_from_annotations(bc_data)

    # Annotate duration of exercises
    exercises = []
    stimes = []
    etimes = []
    duration = []

    df_annot = bc_data.annotations.to_data_frame()

    for event in event_id:
        if "TIMER_START" in event:
            timer, exercise = event.split(' ', 1)
            start_index = df_annot.index[df_annot['description'] == event].tolist()
            start_time = bc_data.annotations[start_index].onset
            if event == 'TIMER_START Electrode pops':
                end_index = df_annot.index[df_annot['description'] == 'TIMER_STOP ' + exercise[0:27]].tolist()
            else:
                end_index = df_annot.index[df_annot['description'] == 'EXERCISE_END ' + exercise[0:27]].tolist()
            end_time = bc_data.annotations[end_index].onset
            occ = len(end_time) #len(start_time)
            if occ > 1: # There are some files where there is double TIMER_START for Mental arithmetic and only one EXERCISE_END
                for i in range(occ):
                    exercises.append(exercise)
                    stimes.append(start_time[i])
                    etimes.append(end_time[i])
                    duration.append(end_time[i] - start_time[i])
            else:
                exercises.append(exercise)
                stimes.append(start_time[0])
                etimes.append(end_time[0]) #error
                duration.append(end_time[0] - start_time[0])

    d = {'exercise': exercises, 'start': stimes, 'end': etimes, 'duration': duration}
    df = pd.DataFrame(data=d)

    # Set new annotations
    new_annot = mne.Annotations(onset=df['start'], duration=df['duration'], description=df['exercise'])
    new_bc_data = bc_data.copy().set_annotations(new_annot)

    return new_bc_data