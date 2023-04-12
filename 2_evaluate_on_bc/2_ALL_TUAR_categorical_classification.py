import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split

from code_submission.load_data import load_dataset
from code_submission.models.EEGModels import EEGNet, ShallowConvNet, DeepConvNet
from code_submission.model_evaluation import model_evaluation_bin, model_evaluation_onehot2, model_evaluation_onehot
from code_submission.models.model_DSCNN import DS_CNNet

model_type = 'DSCNN'
#model_type = 'EEGNet'
#model_type = 'XGBoost'

# Define experiment parameters
N = 290 # 290 ALL
duration = 4
overlap = duration*0.5 # 50% overlap
bin = False
e = 200

#Load train dataset -> TUAR
X, y, _, classes_tuar, class_weights_tuar, _ = load_dataset('TUAR', bin, N, duration, overlap)
print("Whole dataset: Mean:", np.mean(X), " and STD: ", np.std(X))

kernels, chans, samples = 1, 21, 1024 # change according to duration
X_train = X.reshape(X.shape[0], chans, samples, kernels)
y_train = y
print("Reshaped train:", X_train.shape, y_train.shape)

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
fittedModel = model.fit(X_train, y_train, batch_size = 256, epochs = e, verbose = 2, validation_split = 0.2, class_weight=class_weights_tuar) # validation_split=0.125

# save model and architecture to single file
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/without_reftoavg/'
model_name = model_type+'_categorical_'+str(duration)+'s_'+str(e)+'epochs_dropout.h5'
#model.save(path+model_name)

# Plot training metrics
pd.DataFrame(fittedModel.history).plot(figsize=(8,5))
#plt.savefig(path+model_type+'_cat_'+str(duration)+'s_model_history_'+str(e)+'epochs_dropout.png')
#plt.show()
