import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from matplotlib import pyplot

from code_submission.model_evaluation import model_evaluation_bin, model_evaluation_onehot2
from code_submission.load_data import load_dataset

model_type = 'XGBoost'

# Define experiment parameters
N = 290 # 290 ALL
duration = 2
overlap = duration*0.5 # 50% overlap
bin = True

#Load train dataset -> TUAR
X, y, _, classes_tuar, class_weights_tuar = load_dataset('TUAR', bin, N, duration, overlap)
print("Whole dataset: Mean:", np.mean(X), " and STD: ", np.std(X))

print("X: ", X.shape)

# Flatten data to have (#epochs, #channels * #samples)
x_mod = []
for x in X:
    x_mod.append(np.concatenate(x.T))

print("x_mod:", len(x_mod))
X_mod = np.array(x_mod)
print("X_mod: ", X_mod.shape)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X_mod, y, test_size = 0.2, random_state = 42)
print("Split train, validation, test:", X_train.shape, X_test.shape,  y_train.shape, y_test.shape) # (#trials, #channels * #samples)

# Split train into train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# define the datasets to evaluate each iteration
evalset = [(X_train, y_train), (X_val,y_val)]

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train, eval_metric='logloss', eval_set=evalset)

# save model and architecture to single file
path = '/Users/bertavinas/Documents/DTU/Thesis/Code/final_results/models/binary_v2/' # Change path accordingly
model_name = model_type+'_binary.json'
#model.save_model(path+model_name)

# Plot model history
# retrieve performance metrics
results = model.evals_result()
# plot learning curves
pyplot.plot(results['validation_0']['logloss'], label='loss')
pyplot.plot(results['validation_1']['logloss'], label='val_loss')
# show the legend
pyplot.legend()
# show the legend
pyplot.legend()
pyplot.savefig(path+model_type+'_cat_model_history.png')

# make predictions for test data
probs = model.predict(X_test)
y_pred = np.array(probs)

# Return confusion matrix and scores
conf_matrix, scores = model_evaluation_onehot2(y_pred, y_test) # shouldn't be onehot???
#scores.to_csv(path+model_type+'_scores.csv')

fig, ax = plt.subplots(figsize=(10, 8))
sn.heatmap(conf_matrix, annot=True, cmap='Blues', fmt="d", ax=ax)

ax.set_title('Confusion Matrix');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False', 'True'])
ax.yaxis.set_ticklabels(['False', 'True'])

#plt.savefig(path+model_type+'_conf_matrix.png')
#plt.show()