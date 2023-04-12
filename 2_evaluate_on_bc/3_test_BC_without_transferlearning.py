import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import keras

from code_submission.load_data import load_dataset
from code_submission.model_evaluation import model_evaluation_onehot

# Define channels to use
use_chan = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
use_chan_bc = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz', 'R1', 'R2']

# Define experiment parameters
duration = 2
overlap = duration*0.5 # 50% overlap
bin = False
N = 1 # not considered for BC

#Load test dataset -> BrainCapture
X, y, _, classes_bc, class_weights_bc, _ = load_dataset('BC', bin, N, duration, overlap)
print("Whole dataset: Mean:", np.mean(X), " and STD: ", np.std(X))
print(len(y))

kernels, chans, samples = 1, 21, 512
X_test= X.reshape(X.shape[0], chans, samples, kernels)
y_test = y
print("Reshaped train:", X_test.shape, y_test.shape)

# Prepare model
# Load pretrained model from file
model_type = 'DSCNN'
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/code_submission/models/' # change path accordingly
model_name = 'EEGNet_categorical_200epochs_dropout.h5'
pre_trained = keras.models.load_model(path+model_name)
pre_trained.summary()

"""
    Test model
"""
# Make prediction on test set.
probs = pre_trained.predict(X_test)
probs = np.array(probs)

# Evaluate model
conf_matrix, scores = model_evaluation_onehot(probs, y_test, classes_bc)
#scores.to_csv(path+model_type+'_withoutTL_cat_scores.csv')

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10, 8))

sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="d", ax=ax)

ax.set_title('Confusion Matrix', weight='bold', fontsize=23)
ax.set_xlabel('\nPredicted Values', weight='bold', fontsize=18)
ax.set_ylabel('Actual Values ', weight='bold', fontsize=18)

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(classes_bc, fontsize=18)
ax.yaxis.set_ticklabels(classes_bc, fontsize=18)

#fig.savefig(path+model_type+'_cat_BC_conf_matrix_withoutTL.pdf', bbox_inches='tight')
plt.show()