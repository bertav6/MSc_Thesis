import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split

from code_submission.models.model_DSCNN import DS_CNNet
from code_submission.load_data import load_dataset
from code_submission.models.EEGModels import EEGNet
from code_submission.model_evaluation import model_evaluation_onehot2

# Define type of model
#model_type = 'DSCNN'
model_type = 'EEGNet'
#model_type = 'XGBoost'

# Define experiment parameters
N = 290 # 290 ALL
duration = 2
overlap = duration*0.5 # 50% overlap
bin = True
e = 200

#Load train dataset -> TUAR
X, y, _, classes_tuar, class_weights_tuar, _ = load_dataset('TUAR', bin, N, duration, overlap)
print("Whole dataset: Mean:", np.mean(X), " and STD: ", np.std(X))

kernels, chans, samples = 1, 21, 512
X_train = X.reshape(X.shape[0], chans, samples, kernels)
print("Reshaped train:", X_train.shape)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print("Split train, validation, test:", X_train.shape, X_test.shape,  y_train.shape, y_test.shape)

unique_train, counts_train = np.unique(y_train, axis=0,  return_counts=True)
#print("Counts train: ", dict(zip(unique_train, counts_train)))
print("Counts train: ", {tuple(i):j for i,j in zip(unique_train,counts_train)})

unique_test, counts_test = np.unique(y_test, axis=0, return_counts=True)
print("Counts test: ", {tuple(i):j for i,j in zip(unique_test,counts_test)})

"""
    Define model
"""
# Model
if model_type == 'DSCNN':
    model = DS_CNNet(nb_classes=2, Chans=chans, Samples=samples) # samples
elif model_type == 'EEGNet':
    model = EEGNet(nb_classes=2, Chans=chans, Samples=samples,
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
fittedModel = model.fit(X_train, y_train, batch_size = 256, epochs = e, verbose = 2, validation_split = 0.2)#, class_weight=class_weights_tuar)

# save model and architecture to single file
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/models/binary_v2/' # Change path according to directory
model_name = model_type+'_binary_'+str(e)+'epochs.h5'
#model.save(path+model_name)

# Plot training metrics
pd.DataFrame(fittedModel.history).plot(figsize=(8,5))
#plt.savefig(path+model_type+'_model_history_'+str(e)+'epochs.png')
#plt.show()

"""
    Test model
"""
# Make prediction on test set.
probs = model.predict(X_test)
probs = np.array(probs)

# Evaluate model
conf_matrix, scores = model_evaluation_onehot2(probs, y_test)
#scores.to_csv(path+model_type+'_scores_'+str(e)+'epochs.csv')

fig, ax = plt.subplots(figsize=(10, 8))
sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="d", ax=ax)

# Evaluate model
conf_matrix, scores = model_evaluation_onehot2(probs, y_test)
ax.set_title('Confusion Matrix');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

#plt.savefig(path+model_type+'_conf_matrix_'+str(e)+'epochs.png')
#plt.show()


