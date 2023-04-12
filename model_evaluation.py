from sklearn.metrics import fbeta_score, f1_score, recall_score, confusion_matrix, accuracy_score, precision_score
import tensorflow as tf
import numpy as np

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def model_evaluation_onehot(probs, y_test, classes):
    y_pred = probs.argmax(axis=-1)
    #y_pred = probs
    y_true = y_test.argmax(axis=-1)
    #y_true = y_test
    acc = np.mean(y_pred == y_true)
    print("Classification accuracy: %f " % (acc))

    unique, counts = np.unique(y_true, return_counts=True)
    print("Counts test: ", dict(zip(unique, counts)))
    unique, counts = np.unique(y_pred, return_counts=True)
    print("Counts pred: ", dict(zip(unique, counts)))

    # Confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sn.heatmap(cf_matrix, annot=True, cmap='Blues', fmt="d", ax=ax)

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    ## Display the visualization of the Confusion Matrix.
    #plt.show()

    acc = accuracy_score(y_true, y_pred)
    #print("Accuracy sklearn: ", acc)

    recall = recall_score(y_true, y_pred, average=None)
    #print("Recall sklearn: ", recall)

    FP = cf_matrix.sum(axis=0) - np.diag(cf_matrix)
    FN = cf_matrix.sum(axis=1) - np.diag(cf_matrix)
    TP = np.diag(cf_matrix)
    TN = cf_matrix.sum() - (FP + FN + TP)
    # Fall out or false positive rate
    #FPR = FP / (FP + TN)
    # False negative rate -> # False positive rate in our binary classification -> non-art classified as art
    FNR = FN / (TP + FN)
    #print("False Positive Rate: ", FPR) # See onlny FPR of none
    # Specificity or true negative rate # Recall -> art classified as art
    TNR = TN / (TN + FP)
    #print("Recall: ", TNR[4])
    #print("False Posivite Rate: ", FNR[4]) # False positive rate in our binary classification -> non-art classified as art

    pre = precision_score(y_true, y_pred, average=None)
    #print("Precision sklearn: ", pre)

    f1 = f1_score(y_true, y_pred, average='macro')
    #print("F1-score: ", f1)

    wf1 = f1_score(y_true, y_pred, average='weighted')
    print("weighted F1", wf1)

    f2 = fbeta_score(y_true, y_pred, beta=2, average='macro')
    #print("F2-score: ", f2)

    columns = ['Acc', 'Bin Recall', 'Bin FPR', 'WF1-score', 'F1-score', 'F2-score', 'Recall-chew', 'Recall-elec', 'Recall-eyem', 'Recall-musc', 'Recall-none', 'Recall-shiv']
    data = [acc, TNR[4], FNR[4], wf1, f1, f2, recall[0], recall[1], recall[2], recall[3], recall[4], recall[5]]
    new_data = [round(elem*100, 2) for elem in data]
    scores_dict = dict(zip(columns, new_data))
    scores = pd.DataFrame(data=scores_dict, index=[0])
    
    return cf_matrix, scores


def model_evaluation_onehot2(probs, y_test):
    #y_pred = np.rint(probs)
    #print(y_pred.shape, y_pred)
    y_pred = probs.argmax(axis=-1)
    y_test = y_test.argmax(axis=-1)
    acc = np.mean(y_pred == y_test)

    recall = recall_score(y_test, y_pred, average=None)

    f1 = f1_score(y_test, y_pred, average='macro')

    wf1 = f1_score(y_test, y_pred, average='weighted')

    f2 = fbeta_score(y_test, y_pred, beta=2, average='macro')

    unique, counts = np.unique(y_test, return_counts=True)
    print("Counts test: ", dict(zip(unique, counts)))
    unique, counts = np.unique(y_pred, return_counts=True)
    print("Counts pred: ", dict(zip(unique, counts)))

    # Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    FP = cf_matrix[0][1]
    FN = cf_matrix[1][0]
    TP = cf_matrix[1][1]
    TN = cf_matrix[0][0]

    # Fall out or false positive rate
    FPR = FP / (FP + TN)

    columns = ['Acc', 'WF1', 'F1-score', 'F2-score', 'Recall Non-artifact', 'Recall artifact', 'FPR']
    data = [acc, wf1, f1, f2, recall[0], recall[1], FPR]
    new_data = [round(elem * 100, 2) for elem in data]
    scores_dict = dict(zip(columns, new_data))
    scores = pd.DataFrame(data=scores_dict, index=[0])

    return cf_matrix, scores

def model_evaluation_bin(probs, y_test):
    #y_pred = np.rint(probs)
    y_pred = probs.argmax(axis=-1)
    y_test = y_test.argmax(axis=-1)
    #print(y_pred.shape, y_pred)
    acc = np.mean(y_pred == y_test)
    print("Classification accuracy: %f " % (acc))

    unique, counts = np.unique(y_test, return_counts=True)
    print("Counts test: ", dict(zip(unique, counts)))
    unique, counts = np.unique(y_pred, return_counts=True)
    print("Counts pred: ", dict(zip(unique, counts)))

    # Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    print(cf_matrix)
    ax = sn.heatmap(cf_matrix, annot=True, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()