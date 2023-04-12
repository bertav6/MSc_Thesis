import numpy as np
import glob
import time
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.utils import to_categorical
from code_submission.model_evaluation import model_evaluation_onehot

from code_submission.load_data import load_BC_data
import pandas as pd

"""
    Functions
"""

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

    # Add three more layers -> WHY?? Is it necessary if they are the same as pretrained model?
    block2 = transfer_model.layers[12].output  # output of last trained layers
    # block2 = pre_trained.layers[12].output # EEGNet

    flatten = Flatten(name='flatten')(block2)
    dense = Dense(6, activation='softmax')(flatten)

    model = Model(inputs=transfer_model.input, outputs=dense)
    #trans_model.summary()

    #model = Sequential()
    #model.add(trans_model)

    # Compile the model and set the optimizers
    optimizer = keras.optimizers.Adam(lr=0.001)  # missing tf. at the begining
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    #model.summary()

    return model

# load DSCNN
model_path = '/Users/bertavinas/Documents/DTU/Thesis/Code/code_submission/models/'
model_type = 'DSCNN'
model_name = model_path+'DSCNN_categorical_4s_200epochs_dropout.h5'

# retrain on 9 BC files
new_model = create_model(model_name, model_type)

# load BC data
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/BrainCapture/'
bc_files = glob.glob(path+'*_User??_BC*.edf')
# Remove files with different annotations
bc_files.remove(path+'20220823_User00_BC2_P0209.edf') # different exercices' annotation
bc_files.remove(path+'20220823_User01_BC5_P0209.edf') # different exercices' annotation
bc_files.remove(path+'20220927_User10_BC2_P0188.edf') # ECG leakage
bc_files.remove(path+'20221018_User14_BC2_P0209.edf') # noisy channels + ECG leakage
bc_files.remove(path+'20221027_User20_BC2_P0209.edf') # bad results on CV
bc_files.remove(path+'20221019_User16_BC2_P0209.edf') # ECG leakage
# Remove file used for testing
bc_files.remove(path+'20221024_User18_BC2_P0209.edf')
bc_files.sort()
train_files = bc_files
test_file = ['/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/BrainCapture/20221024_User18_BC2_P0209.edf']

print("Train BC files: ", len(train_files))
print("Test BC files: ", len(test_file))

# Timer
time0 = time.time()

# Get data for all BC files
# TRAIN
use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'F9', 'F10']
X1, y1, _ = load_BC_data(train_files, use_chan_bc, binary=False, e_duration=4, e_overlap=4*0.5)
X2 = np.concatenate(X1)
y = np.concatenate(y1)

kernels, chans, samples = 1, 21, 1024
X_train = X2.reshape(X2.shape[0], chans, samples, kernels) # add final dimension kernels!
print("Reshaped train:", X_train.shape)

# Check number of samples and classes
unique, counts = np.unique(y, return_counts=True)
print("Counts train: ", dict(zip(unique, counts)))
# One hot encoding
label_encoder = LabelEncoder()
y_onehot = label_encoder.fit_transform(y)
y_train = to_categorical(y_onehot)

classes = unique
# Class weights
y_integers = np.argmax(y_train, axis=1)
weights = class_weight.compute_class_weight(class_weight="balanced", classes = np.unique(y_integers) , y = y_integers)
class_weights = dict(zip(np.unique(y_integers), weights))
print(class_weights)

#Shuffle to avoid learning temporal patterns
p = np.random.RandomState(seed=15).permutation(X_train.shape[0])
data = X_train[p]
labels = y_train[p]

# Timer
time1 = time.time()
print("Time to load 9 BC files: ", time1-time0)

# Fit the model
fittedModel = new_model.fit(data, labels, batch_size=256, epochs=50, verbose=0, class_weight=class_weights)
new_model.save(model_path+'retrained_9BC_50epochs.h5')

# Timer
time2 = time.time()
print("Time to train DSCNN model with 9 BC files and 50 epochs: ", time2-time1)

# TEST
X3, y3, _ = load_BC_data(test_file, use_chan_bc, binary=False, e_duration=4, e_overlap=4*0.5)
X4 = np.concatenate(X3)
y4 = np.concatenate(y3)
label_encoder = LabelEncoder()
y_onehot = label_encoder.fit_transform(y4)
y_test = to_categorical(y_onehot)

# Timer
time3 = time.time()
print("Time to load test 1 BC file: ", time3-time2)

X_test = X4.reshape(X4.shape[0], chans, samples, kernels) # add final dimension kernels!
print("Reshaped train:", X_test.shape)

# Make prediction on test set.
probs = new_model.predict(X_test)
probs = np.array(probs)
#pred_labels = np.argmax(probs, axis=1)
#print(pred_labels)

# Random classifier
import random
freqs = list(dict(zip(unique, counts)).values())
weights = [x / X_train.shape[0] for x in freqs]

# Counts train:  {'chew': 560, 'elec': 722, 'eyem': 561, 'musc': 285, 'none': 4700, 'shiv': 279} / 7107
numberList = [0, 1, 2, 3, 4, 5]
rand_pred = random.choices(numberList, weights=weights, k=len(probs))

# Evaluate model
conf_matrix, scores_normal = model_evaluation_onehot(probs.argmax(axis=-1), y_test.argmax(axis=-1), classes)
conf_matrix_random, scores_random = model_evaluation_onehot(rand_pred, y_test.argmax(axis=-1), classes)

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
fig.savefig(path + model_type + '_TL_User18.pdf', bbox_inches='tight')

# The scope of these changes made to
# pandas settings are local to with statement.
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(scores_normal)
    print(scores_random)

# Timer
time4 = time.time()
print("Time to predict 1 BC file: ", time4-time3)

print("TOTAL TIME: ", time4-time0)
#print(pred_labels)


