import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code_submission.load_data import load_BC_data, load_TUAR_data
import glob

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from keras.utils import to_categorical

import keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten

from keras.callbacks import LambdaCallback
import shap

def create_model(model_name, model_type):
    transfer_model = 0
    transfer_model = keras.models.load_model(model_name)
    transfer_model.summary()
    #caca

    # see initial weights
    initial_weights = transfer_model.layers[-1].get_weights()[0]
    print("Initial models when model is loaded: ", transfer_model.layers[-1].weights)

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
    dense = Dense(len(classes), activation='softmax', kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42))(flatten)
    model = Model(inputs=transfer_model.input, outputs=dense)

    # see initial weights
    #print("New model before retraining: ", model.layers[-1].weights)

    #trans_model.summary()

    #model = Sequential()
    #model.add(trans_model)

    # Compile the model and set the optimizers
    optimizer = keras.optimizers.Adam(lr=0.001)  # missing tf. at the begining
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    #model.summary()

    return model, initial_weights

# Define channels to useplt.savefig('books_read.png')
use_chan = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']#, 'EEG T1-REF', 'EEG T2-REF']
use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz']#, 'F9', 'F10']

# Define experiment parameters
duration = 4
overlap = duration*0.50 # 50% overlap
bin = False

# Load BC data
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/Datasets/BrainCapture/'
bc_files = glob.glob(path+'*_User??_BC*.edf')

# Remove files with different annotations
bc_files.remove(path+'20220823_User00_BC2_P0209.edf') # different exercices' annotation
bc_files.remove(path+'20220823_User01_BC5_P0209.edf') # different exercices' annotation
bc_files.remove(path+'20220927_User10_BC2_P0188.edf') # ECG leakage
bc_files.remove(path+'20221018_User14_BC2_P0209.edf') # noisy channels + ECG leakage
bc_files.remove(path+'20221027_User20_BC2_P0209.edf') # bad results on CV
bc_files.remove(path+'20221019_User16_BC2_P0209.edf') # ECG leakage
bc_files.sort()

#bc_files.remove(path+'20220927_User11_BC2_P0188.edf') # noisy channels + ECG leakage
print("BC files: ", len(bc_files))
print(bc_files)

# Get data for all BC files
X_bc, y_bc, users_id = load_BC_data(bc_files, use_chan_bc, binary=bin, e_duration=duration, e_overlap=overlap)
X1 = np.concatenate(X_bc)
y = np.concatenate(y_bc)
print("shape: ", X1.shape)

kernels, chans, samples = 1, 21, 1024
X = X1.reshape(X1.shape[0], chans, samples, kernels) # add final dimension kernels!
print("Reshaped train:", X.shape)

print("Users id:", users_id)
print("Whole dataset: Mean:", np.mean(X), " and STD: ", np.std(X))

groups = []
i = 0
for sub in X_bc:
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

results_path = path
e = 1

print("Train samples: ", X.shape, y.shape)
k_fold_model, initial_weights = create_model(model_name, model_type)

#Shuffle to avoid learning temporal patterns
p = np.random.RandomState(seed=15).permutation(X.shape[0])

# Keep track of weights
weights_dict = {}
weight_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: weights_dict.update({epoch:k_fold_model.layers[-1].get_weights()[0]}))

# Fit the model
fittedModel = k_fold_model.fit(X[p], y[p], batch_size=256, epochs=e, verbose=0, class_weight=class_weights, validation_split = 0.2, callbacks=[weight_callback])#callbacks=weight_callback)
#pd.DataFrame(fittedModel.history).plot(figsize=(8,5))
plt.rcParams.update({'font.size': 16})
#plt.savefig(path+model_type+'_WEIGHTS_cat_model_history_'+str(e)+'epochs.pdf', bbox_inches='tight')

final_weights = k_fold_model.layers[-1].get_weights()[0]

# plot matrices
fig, axs = plt.subplots(2, figsize=(15, 7))
# find minimum of minima & maximum of maxima
minmin = np.min([np.min(initial_weights), np.min(final_weights)])
maxmax = np.max([np.max(initial_weights), np.max(final_weights)])
print(initial_weights.shape, final_weights.shape)

im0 = axs[0].imshow(initial_weights.T, vmin=minmin, vmax=maxmax, aspect="auto")
axs[0].set_title("Weights matrix before transfer learning", fontsize = 25)
axs[0].axis('off')
axs[0].set_aspect("equal")
im1 = axs[1].imshow(final_weights.T, vmin=minmin, vmax=maxmax, aspect="auto")
axs[1].set_title("Weights matrix after transfer learning", fontsize = 25)
axs[1].axis('off')
axs[1].set_aspect("equal")

im_ratio = final_weights.T.shape[1]/final_weights.T.shape[0]
fig.colorbar(im0, orientation="horizontal")
#plt.savefig("weights_matrix.pdf", bbox_inches = 'tight', fraction=0.047*im_ratio)
plt.show()