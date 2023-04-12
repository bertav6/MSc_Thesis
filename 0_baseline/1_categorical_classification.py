import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import seaborn as sn
from sklearn.model_selection import train_test_split

from arl_eegmodels_master.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from code_submission.model_evaluation import model_evaluation_onehot
from code_submission.load_data import load_dataset
from code_submission.models.model_DSCNN import DS_CNNet

#model_type = 'DSCNN'
model_type = 'EEGNet'
#model_type = 'XGBoost'

# Define experiment parameters
N = 290 # 290 ALL
duration = 2
overlap = duration*0.5 # 50% overlap
bin = False
e = 200

#Load train dataset -> TUAR
Xi, y, _, classes_tuar, class_weights_tuar, _ = load_dataset('TUAR', bin, N, duration, overlap)
print("Whole dataset: Mean:", np.mean(Xi), " and STD: ", np.std(Xi))

kernels, chans, samples = 1, 21, 512
X = Xi.reshape(Xi.shape[0], chans, samples, kernels) # add final dimension kernels!
print("Reshaped train:", X.shape)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Split train, validation, test:", X_train.shape, X_test.shape,  y_train.shape, y_test.shape)

y_tr = y_train.argmax(axis=-1)
y_te = y_test.argmax(axis=-1)

unique, counts = np.unique(y_tr, return_counts=True)
print("Counts train: ", dict(zip(unique, counts)))
unique, counts = np.unique(y_te, return_counts=True)
print("Counts test: ", dict(zip(unique, counts)))

"""
    Define model
"""

# Model
if model_type == 'DSCNN':
    model = DS_CNNet(nb_classes=len(classes_tuar), Chans=chans, Samples=samples) # samples
elif model_type == 'EEGNet':
    model = EEGNet(nb_classes=len(classes_tuar), Chans=chans, Samples=samples,
                   dropoutRate=0.5, kernLength=256, F1=8, D=2, F2=16,  # kernellength = 32
                   dropoutType='Dropout')
model.summary()

"""
    Train model
"""

# Compile the model and set the optimizers
optimizer = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Fit the model
fittedModel = model.fit(X_train, y_train, batch_size = 256, epochs = e, verbose = 2, validation_split = 0.2, class_weight=class_weights_tuar)

# save model and architecture to single file
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/' # change path accordingly
model_name = model_type+'_categorical_TUAR_'+str(e)+'epochs.h5'
#model.save(path+model_name)

# Plot training metrics
pd.DataFrame(fittedModel.history).plot(figsize=(8,5))
plt.savefig(path+model_type+'_cat_TUAR_model_history_'+str(e)+'epochs.png')
#plt.show()

"""
    Load pretrained model
"""
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/models/categorical_v2/'
model_name = 'EEGNet_categorical_200epochs.h5'
model = keras.models.load_model(path+model_name)
model.summary()

"""
    Test model
"""
# Make prediction on test set.
probs = model.predict(X_test)
probs = np.array(probs)

# Evaluate model
conf_matrix, scores = model_evaluation_onehot(probs, y_test, classes_tuar)
#scores.to_csv(path+model_type+'_cat_scores_'+str(e)+'epochs.csv')

plt.rcParams.update({'font.size': 16})

fig, ax = plt.subplots(figsize=(10, 8))

sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="d", ax=ax)

ax.set_title('Confusion Matrix', weight='bold', fontsize=23)
ax.set_xlabel('\nPredicted Values', weight='bold', fontsize=18)
ax.set_ylabel('Actual Values ', weight='bold', fontsize=18)

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(classes_tuar, fontsize=18)
ax.yaxis.set_ticklabels(classes_tuar, fontsize=18)

#fig.savefig(path+model_type+'_cat_TUAR_conf_matrix_'+str(e)+'epochs.pdf', bbox_inches='tight')
#plt.show()


