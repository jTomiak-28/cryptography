# Use sklearn to classify the plaintext recognition data with logit

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys


def train():
    
    # define model
    model = LogisticRegression(C=10, penalty='l2')
    
    # build dataframes
    train_frame = pd.read_csv("data/train.csv", header=None)
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


def tune():
    
    # define hyperparameter choices
    param_grid = {
        'C': [1000, 100, 10, 1.0, 0.1, 0.01, 0.001], 
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    # define learning model and tuning process
    logit = LogisticRegression()
    
    tuned_model = GridSearchCV(logit, param_grid=param_grid)
    
    # build dataframes (using just 5000 from train set for tuning)
    train_frame = pd.read_csv("data/train.csv", nrows=5000, header=None)
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
    df.to_csv("results/logit_tuning_results.csv", index=False)

train()