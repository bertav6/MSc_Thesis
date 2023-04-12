import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from code_submission.load_data import load_BC_data, load_TUAR_data, load_FIL_data
import glob
import seaborn as sn

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.utils import to_categorical
from code_submission.model_evaluation import model_evaluation_onehot

import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten

import wandb
from wandb.keras import WandbCallback

def create_model(model_name, model_type):
    transfer_model = 0
    transfer_model = keras.models.load_model(model_name)
    #transfer_model.summary()

    # Freeze weights in first layers
    if model_type == 'DSCNN':
        for layer in transfer_model.layers[0:13]:  # From Input to AveragePool2D (including SepConv2D)
            layer.trainable = False
    elif model_type == 'EEGNet':
        for layer in transfer_model.layers[0:13]: #EEGNet
            layer.trainable = False

    # Add three more layers
    block2 = transfer_model.layers[12].output  # output of last trained layers
    # block2 = pre_trained.layers[12].output # EEGNet

    flatten = Flatten(name='flatten')(block2)
    dense = Dense(len(classes), activation='softmax')(flatten)

    model = Model(inputs=transfer_model.input, outputs=dense)
    #trans_model.summary()

    #model = Sequential()
    #model.add(trans_model)

    # Compile the model and set the optimizers
    optimizer = keras.optimizers.Adam(lr=0.001)  # missing tf. at the begining
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    #model.summary()

    return model

# Define channels to useplt.savefig('books_read.png')
use_chan = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']#, 'EEG T1-REF', 'EEG T2-REF']
use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'R1', 'R2']
use_chan_fil = ['EEG Fp1-REF', 'EEG Fp2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
                    'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T7-REF', 'EEG T8-REF', 'EEG P7-REF',
                    'EEG P8-REF', 'EEG T9-REF', 'EEG T10-REF', 'EEG Fz-REF', 'EEG Cz-REF', 'EEG Pz-REF']  # , 'EEG F10-REF', 'EEG F9-REF', 'EEG P9-REF', 'ECG EKG-REF', 'EEG P10-REF', 'EEG TP7-REF', 'EEG TP8-REF', 'Photic-REF', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']

# Define experiment parameters
duration = 4
overlap = duration*0.50 # 50% overlap
bin = False

path = '/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/BrainCapture/'
fil_files = glob.glob(path + '*_User??_FIL*.edf')
bc_files = glob.glob(path + '*_User??_BC*.edf')

# Remove files with different annotations
fil_files.remove(path + '20220823_User00_FIL2_P0209.edf')  # different exercices' annotation
fil_files.remove(path + '20220823_User01_FIL5_P0209.edf')  # different exercices' annotation
fil_files.remove(path + '20220927_User10_FIL2_P0188.edf')  # ECG leakage
fil_files.remove(path + '20221018_User14_FIL2_P0209.edf')  # noisy channels + ECG leakage
fil_files.remove(path + '20221027_User20_FIL2_P0209.edf') # bad results on CV
fil_files.remove(path + '20221019_User16_FIL2_P0209.edf') # ECG leakage

# Remove files with different annotations
bc_files.remove(path + '20220823_User00_BC2_P0209.edf')  # different exercices' annotation
bc_files.remove(path + '20220823_User01_BC5_P0209.edf')  # different exercices' annotation
bc_files.remove(path + '20220927_User10_BC2_P0188.edf')  # ECG leakage
bc_files.remove(path + '20221018_User14_BC2_P0209.edf')  # noisy channels + ECG leakage
bc_files.remove(path + '20221027_User20_BC2_P0209.edf') # bad results on CV
bc_files.remove(path + '20221019_User16_BC2_P0209.edf') # ECG leakage

print(len(fil_files), len(bc_files))
bc_files.sort()
fil_files.sort()

# Get data for all FIL files
X_fil, y_fil = load_FIL_data(fil_files, bc_files, use_chan_fil, binary=bin, e_duration=duration, e_overlap=overlap)
X = np.concatenate(X_fil)
y = np.concatenate(y_fil)
print(X.shape, y.shape)
print("Whole dataset: Mean:", np.mean(X), " and STD: ", np.std(X))

groups = []
i = 0
for sub in X_fil:
    print(len(sub))
    groups.append(np.full(len(sub), i))
    i = i+1
groups = np.concatenate(groups).ravel()

# Check number of samples and classes
unique, counts = np.unique(y, return_counts=True)
print("Counts train: ", dict(zip(unique, counts)))

# One hot encoding
label_encoder = LabelEncoder()
y_onehot = label_encoder.fit_transform(y)
y = to_categorical(y_onehot)
classes = unique

# Class weights
y_integers = np.argmax(y, axis=1)
weights = class_weight.compute_class_weight(class_weight="balanced", classes = np.unique(y_integers) , y = y_integers)
class_weights = dict(zip(np.unique(y_integers), weights))
print(class_weights)

# Prepare model
# Load pretrained model from file
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/code_submission/models/'
model_type = 'DSCNN'
#model_type = 'EEGNet'
model_name = path+'DSCNN_categorical_4s_200epochs_dropout.h5'

e = 1
num_folds = 10

# Define the K-fold Cross Validator
logo = LeaveOneGroupOut()

# Define per-fold score containers
scores_fold = []

# K-fold Cross Validation model evaluation
fold_no = 1
print(X.shape, y.shape, groups.shape)

for train, test in logo.split(X, y, groups):
    print("Train samples: ", len(train))
    print("Test samples: ", len(test))

    k_fold_model = create_model(model_name, model_type)
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    #Shuffle to avoid learning temporal patterns
    data = X[train]
    labels = y[train]
    #p = np.random.permutation(len(data))
    p = np.random.RandomState(seed=15).permutation(len(data))

    # Fit the model
    fittedModel = k_fold_model.fit(data[p], labels[p], batch_size=256, epochs=e, verbose=0, class_weight=class_weights)
    #pd.DataFrame(fittedModel.history).plot(figsize=(8, 5))
    #plt.show()

    # Make prediction on test set.
    probs = k_fold_model.predict(X[test])
    probs = np.array(probs)

    # Evaluate model
    conf_matrix, scores = model_evaluation_onehot(probs, y[test], classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="d", ax=ax)

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    figure = ax.get_figure()

    scores_fold.append(scores)

    # Increase fold number
    fold_no = fold_no + 1

final_scores = pd.concat(scores_fold)

# The scope of these changes made to
# pandas settings are local to with statement.
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(final_scores)

#final_scores.to_csv(path+model_type+'_FIL_200epochs_TL_cat_scores_dropout_'+str(e)+'EPOCHS.csv')