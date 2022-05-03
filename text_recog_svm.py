# Use sklearn to classify the plaintext recognition data with svm

# link to pickle tutorial:
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle
import sys


def train():
    
    # define model
    model = SVC(C=1.0, kernel='rbf', gamma='scale')
    
    # build dataframes
    train_frame = pd.read_csv("data/train.csv", header=None, nrows=50000)
    test_frame = pd.read_csv("data/test.csv", header=None)
    
    # select labels and data
    train_data = train_frame.iloc[:, 1:]
    train_labels = train_frame.iloc[:, 0]
    
    test_data = test_frame.iloc[:, 1:]
    test_labels = test_frame.iloc[:, 0]
    
    print('Starting training')
    model.fit(train_data, train_labels)

    # evaluate
    predicted = model.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted)
    f1 = f1_score(test_labels, predicted, average='binary')
    conf_mat = confusion_matrix(test_labels, predicted)

    print("Accuracy: ", accuracy)
    print("F1-score: ", f1)
    print("Confusion matrix:\n", conf_mat)
    
    filename = "models/final_svm.sav"
    pickle.dump(model, open(filename, 'wb')) # save model to file
    
    


def tune():
    
    # define hyperparameter choices
    param_grid = {
        'C': [1.0, 0.1, 0.01, 0.001], 
        'gamma': [1.0, 0.1, 0.01, 0.001, 'scale'],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    svm = SVC()
    
    tuned_model = GridSearchCV(svm, param_grid=param_grid)
    
    # build dataframes (using just 5000 from train set for tuning)
    train_frame = pd.read_csv("data/train.csv", nrows=1000, header=None)
    test_frame = pd.read_csv("data/test.csv", header=None)
    
    # select labels and data
    train_data = train_frame.iloc[:, 1:]
    train_labels = train_frame.iloc[:, 0]

    test_data = test_frame.iloc[:, 1:]
    test_labels = test_frame.iloc[:, 0]
    
    # start training
    print('Starting tuning')
    tuned_model.fit(train_data, train_labels)

    # evaluate
    predicted = tuned_model.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted)
    f1 = f1_score(test_labels, predicted, average='binary')
    
    # print tuning results to file
    df = pd.DataFrame(tuned_model.cv_results_)
    df.to_csv("results/svm_tuning_results.csv", index=False)

train()