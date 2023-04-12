import mne
import pandas as pd
import re
import glob
import numpy as np

from data_preprocessing import data_preprocessing
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

# Load both datasets
def load_dataset(dataset, binary, N, duration, overlap):
    # Define channels to use
    use_chan = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
                'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
                #'EEG T1-REF', 'EEG T2-REF']
    use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                   'T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'F9', 'F10']
    use_chan_fil = ['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T7-REF', 'EEG T8-REF',
                    'EEG P7-REF', 'EEG P8-REF', 'EEG T9-REF', 'EEG T10-REF', 'EEG Fz-REF', 'EEG Cz-REF', 'EEG Pz-REF' ]#, 'EEG F9-REF', 'EEG F10-REF']#, 'EEG P9-REF', 'ECG EKG-REF', 'EEG P10-REF', 'EEG TP7-REF', 'EEG TP8-REF', 'Photic-REF', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']

    users_id = None
    if dataset == 'TUAR':
        # List of .edf files
        file_list = open('/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/TUAR/list/edf_01_tcp_ar.list', "r")
        X, y, event_id = load_TUAR_data(file_list, use_chan, binary=binary, n_sub=N, e_duration=duration, e_overlap=overlap)

        print("Files: ", len(X))
        # SPLIT WITHOUT CONSIDERING SUBJECTS
        X = np.concatenate(X)
        y = np.concatenate(y)
        print("Samples: ", len(X))

        groups = None

    else:

        # Load BC and FIL data
        path = '/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/BrainCapture/'
        bc_files = glob.glob(path + '*_User??_BC*.edf')
        fil_files = glob.glob(path + '*_User??_FIL*.edf')

        # Remove files with different annotations
        bc_files.remove(path + '20220823_User00_BC2_P0209.edf')  # different exercices' annotation
        bc_files.remove(path + '20220823_User01_BC5_P0209.edf')  # different exercices' annotation
        bc_files.remove(path + '20220927_User10_BC2_P0188.edf')  # ECG leakage
        bc_files.remove(path + '20221018_User14_BC2_P0209.edf')  # noisy channels + ECG leakage
        bc_files.remove(path + '20221027_User20_BC2_P0209.edf')  # bad results on CV
        bc_files.remove(path + '20221019_User16_BC2_P0209.edf')  # ECG leakage

        # Remove files with different annotations
        fil_files.remove(path + '20220823_User00_FIL2_P0209.edf')  # different exercices' annotation
        fil_files.remove(path + '20220823_User01_FIL5_P0209.edf')  # different exercices' annotation
        fil_files.remove(path + '20220927_User10_FIL2_P0188.edf')  # ECG leakage
        fil_files.remove(path + '20221018_User14_FIL2_P0209.edf')  # noisy channels + ECG leakage
        #fil_files.remove(path + '20221027_User20_FIL2_P0209.edf')  # bad results on CV
        #fil_files.remove(path + '20221019_User16_FIl2_P0209.edf')  # ECG leakage

        bc_files.sort()
        fil_files.sort()

        if dataset == 'BC':
            # Get data for all BC files
            X_bc, y, users_id = load_BC_data(bc_files, use_chan_bc, binary=binary, e_duration=duration, e_overlap=overlap)

            print("Files: ", len(X_bc))
            # SPLIT WITHOUT CONSIDERING SUBJECTS
            X = np.concatenate(X_bc)
            y = np.concatenate(y)
            print("Samples: ", len(X))

            groups = []
            i = 0
            for sub in X_bc:
                groups.append(np.full(len(sub), i))
                i = i + 1
            groups = np.concatenate(groups).ravel()

        elif dataset == 'FIL':
            # Get data for all BC files
            X_fil, y = load_FIL_data(fil_files, bc_files, use_chan_fil, binary=binary, e_duration=duration, e_overlap=overlap)

            print("Files: ", len(X_fil))
            # SPLIT WITHOUT CONSIDERING SUBJECTS
            X = np.concatenate(X_fil)
            y = np.concatenate(y)
            print("Samples: ", len(X))

            groups = []
            i = 0
            for sub in X_fil:
                groups.append(np.full(len(sub), i))
                i = i + 1
            groups = np.concatenate(groups).ravel()

    unique, counts = np.unique(y, return_counts=True)
    print("Counts: ", dict(zip(unique, counts)))

    # One hot encoding
    label_encoder = LabelEncoder()
    y_onehot = label_encoder.fit_transform(y)
    y = to_categorical(y_onehot)
    classes = unique

    # Class weights
    y_integers = np.argmax(y, axis=1)
    weights = class_weight.compute_class_weight(class_weight="balanced", classes=np.unique(y_integers), y=y_integers)
    class_weights = dict(zip(np.unique(y_integers), weights))
    print(class_weights)

    return X, y, groups, classes, class_weights, users_id


# Load TUAR dataset
def load_TUAR_data(file_list, use_chan, binary, n_sub, e_duration, e_overlap):
    """
    Function lo load TUAR data

    :param file_list: list of .edf files in directory
    :param use_chan: channels to load
    :return: X -> matrix with EEG data
            y -> vector with EEG labels
    """

    content = file_list.readlines()

    # List of annotations
    df_labels = pd.read_csv('/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/TUAR/csv/labels_01_tcp_ar.csv',
                            header=4,
                            skiprows=[6], skipinitialspace=True)
    df_labels['duration'] = df_labels['stop_time'] - df_labels['start_time']

    # Remove TUAR files where there are artifact longer than 30 seconds
    #files_remove = df_labels[df_labels['duration'] > 30]['# key'].unique()

    artifacts = list(df_labels.artifact_label.unique())
    artifacts.append('none')
    values_list = list(range(len(artifacts)))
    event_id = dict(zip(artifacts, values_list))

    # Load data for all files
    raw_sub = []
    labels_sub = []
    for recording in content[0:n_sub]:  # only small dataset
        file = re.findall("/.*.edf$", recording)
        path = '/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/TUAR'
        raw = mne.io.read_raw_edf(path + file[0], preload=True)
        if raw.info['sfreq'] >= 256:
            X, y = data_preprocessing(raw, path + file[0], use_chan, dataset='TUAR', binary=binary, e_duration=e_duration, e_overlap=e_overlap)
            raw_sub.append(X)
            labels_sub.append(y)

    X = raw_sub
    y = labels_sub

    return X, y, event_id

# Load BC dataset
def load_BC_data(file_list, use_chan, binary, e_duration, e_overlap):
    """
    Function lo load TUAR data

    :param file_list: list of .edf files in directory
    :param use_chan: channels to load
    :return: X -> matrix with EEG data
            y -> vector with EEG labels
    """

    # Load data for all files
    raw_sub = []
    labels_sub = []

    users = []
    samples = []
    for file in file_list:  # only small dataset
        raw = mne.io.read_raw_edf(file, preload=True)
        if raw.info['sfreq'] >= 256:
            X, y = data_preprocessing(raw, file, use_chan, dataset='BC', binary=binary, e_duration=e_duration, e_overlap=e_overlap)
            raw_sub.append(X)
            labels_sub.append(y)
            users.append((re.findall(r'User\d{2}',file))[0])
            samples.append(X.shape[0])

    users_id = dict(zip(users, samples))
    X = raw_sub
    y = labels_sub

    return X, y, users_id

# Load BC dataset
def load_FIL_data(file_list, bc_files, use_chan, binary, e_duration, e_overlap):
    """
    Function lo load TUAR data

    :param file_list: list of .edf files in directory
    :param use_chan: channels to load
    :return: X -> matrix with EEG data
            y -> vector with EEG labels
    """

    # Load data for all files
    raw_sub = []
    labels_sub = []

    i = 0
    for file in file_list:  # only small dataset
        raw_fil = mne.io.read_raw_edf(file, preload=True)
        raw_bc = mne.io.read_raw_edf(bc_files[i], preload=True)
        if raw_fil.info['sfreq'] >= 256:
            # Put BC annotations on FIL recording
            new_annot = raw_bc.annotations
            # if BC annotations start before FIL recording
            df = raw_bc.annotations.to_data_frame()
            df_eyes_closed = df[df['description'] == 'TIMER_START Eyes closed']
            import pytz
            utc = pytz.UTC
            first_eyes_closed = df_eyes_closed.iloc[0]['onset'].to_pydatetime().replace(tzinfo=utc)
            fil_start = raw_fil.info['meas_date'].replace(tzinfo=utc)
            if first_eyes_closed < fil_start:
                dif = raw_bc.info['meas_date'] - raw_fil.info['meas_date']
                delay = abs(dif.total_seconds())
                new_annot.append(onset=[delay], duration=[0.0], description=['TIMER_START Eyes closed'])

            new_fil = raw_fil.copy().set_annotations(new_annot)
            #final_fil = new_annotated_bc_data(new_fil)
            #final_fil.plot(block=True)
            i = i + 1

            X, y = data_preprocessing(new_fil, file, use_chan, dataset='FIL', binary=binary, e_duration=e_duration, e_overlap=e_overlap)
            raw_sub.append(X)
            labels_sub.append(y)

    X = raw_sub
    y = labels_sub

    return X, y


