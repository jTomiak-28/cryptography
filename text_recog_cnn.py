# Tune and train a CNN to recognize plaintexts. Based off of (directly copied from)
# the CNN code I wrote for my science fair project.

# path for cmd access:
# C:\Users\Tomiakfamily\Documents\Govschewlwork\2021-2022\Cryptography\Python

import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from numpy import vstack
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

# This module defines the datasets for use in pytorch models in 

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# class to load plaintext recognition dataset
class PTDataset(Dataset):

    def __init__(self, filename):
        """
        Create new pt dataset from filename of data file
        Args:
            filename (str): path to csv data
        """
        self.sms_dataframe = pd.read_csv(filename)
        
    def __len__(self):
        return len(self.sms_dataframe)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        label = self.sms_dataframe.iloc[index, 0]
        
        # use identity matrix to create one-hot encodings for letters, with extra col
        # of all zeros on the end for padding value 26
        char_array = np.identity(26)
        padding_array = np.zeros((26,1))
        char_array = np.hstack((char_array,padding_array))
        
        text = self.sms_dataframe.iloc[index, 1:]
        
        data = np.array([char_array[:,char] for char in list(text)], dtype=np.float32)
        
        return data, label


# CNN for classifying plaintext data
class CNN(nn.Module):
    # set architecture
    def __init__(self, config):
        super(CNN, self).__init__()
        
        # initialize attributes needed for forward()
        self.conv_layers = config["conv_layers"]
        self.linear_layers = config["linear_layers"]
        self.dropout = config["dropout"]
        
        # first hidden layer (conv), 20 is the length of each input
        self.hidden1 = nn.Conv1d(20, config["num_filters"], config["filter_size"])
                                # num input channels, num out channels (num filters), size of each filter
        if config["init"] == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        elif config["init"] == 'kaiming_normal':
            nn.init.kaiming_normal_(self.hidden1.weight, nonlinearity='relu')
        elif config["init"] == 'xavier_uniform':
            nn.init.xavier_uniform_(self.hidden1.weight)
        elif config["init"] == 'xavier_normal':
            nn.init.xavier_normal_(self.hidden1.weight)
        self.act1 = nn.ReLU()
        
        # first pooling layer
        self.pool1 = nn.MaxPool1d(2) # size of window should be fine at 2, halves input
        
        # second conv layer?
        if config["conv_layers"] == 2:
            self.hidden1b = nn.Conv1d(config["num_filters"], config["num_filters"], config["filter_size"])
                                # num input channels, num out channels (num filters), size of each filter
            if config["init"] == 'kaiming_uniform':
                nn.init.kaiming_uniform_(self.hidden1b.weight, nonlinearity='relu')
            elif config["init"] == 'kaiming_normal':
                nn.init.kaiming_normal_(self.hidden1b.weight, nonlinearity='relu')
            elif config["init"] == 'xavier_uniform':
                nn.init.xavier_uniform_(self.hidden1b.weight)
            elif config["init"] == 'xavier_normal':
                nn.init.xavier_normal_(self.hidden1b.weight)
            self.act1b = nn.ReLU()
            
            # second pooling layer
            self.pool1b = nn.MaxPool1d(2) # size of window should be fine at 2, halves input
        
        if config["dropout"] != 0:
            self.dropout_layer = nn.Dropout(config["dropout"])
        
        linear_input = int((26-(config["filter_size"]-1))/2)*config["num_filters"]
        if config["conv_layers"] == 2:
            linear_input = int(((linear_input/config["num_filters"])-(config["filter_size"]-1))/2)*config["num_filters"]
         
        # fully connected layer
        self.hidden2 = nn.Linear(linear_input, config["linear_size"])
        # Input layer has 26 channels (possible letters), conv layer sees
        # filter_size - 1 layers cut off since there's no padding, pooling halves and
        # rounds down.  So that's how the input size to the linear layer is calculated.
        if config["init"] == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        elif config["init"] == 'kaiming_normal':
            nn.init.kaiming_normal_(self.hidden2.weight, nonlinearity='relu')
        elif config["init"] == 'xavier_uniform':
            nn.init.xavier_uniform_(self.hidden2.weight)
        elif config["init"] == 'xavier_normal':
            nn.init.xavier_normal_(self.hidden2.weight)
        self.act2 = nn.ReLU()
        
        # second linear layer?
        if config["linear_layers"] == 2:
            self.hidden2b = nn.Linear(config["linear_size"], config["linear_size"])
            if config["init"] == 'kaiming_uniform':
                nn.init.kaiming_uniform_(self.hidden2b.weight, nonlinearity='relu')
            elif config["init"] == 'kaiming_normal':
                nn.init.kaiming_normal_(self.hidden2b.weight, nonlinearity='relu')
            elif config["init"] == 'xavier_uniform':
                nn.init.xavier_uniform_(self.hidden2b.weight)
            elif config["init"] == 'xavier_normal':
                nn.init.xavier_normal_(self.hidden2b.weight)
            self.act2b = nn.ReLU()
                
        # output layer
        self.hidden3 = nn.Linear(config["linear_size"], 2)
        if config["init"] == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        elif config["init"] == 'kaiming_normal':
            nn.init.kaiming_normal_(self.hidden3.weight, nonlinearity='relu')
        elif config["init"] == 'xavier_uniform':
            nn.init.xavier_uniform_(self.hidden3.weight)
        elif config["init"] == 'xavier_normal':
            nn.init.xavier_normal_(self.hidden3.weight)
        self.act3 = nn.Softmax(dim=1)
        
    def forward(self, X):
        
        # first conv+pool layer
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        
        # possible second conv+pool layer
        if self.conv_layers == 2:
            X = self.hidden1b(X)
            X = self.act1b(X)
            X = self.pool1b(X)   
        
        # dropout if required
        if self.dropout != 0:
            X = self.dropout_layer(X)
        
        # flatten
        X = torch.flatten(X, 1)
        
        # first fully connected layer
        X = self.hidden2(X)
        X = self.act2(X)
        
        # optional second fully connected layer
        if self.linear_layers == 2:
            X = self.hidden2b(X)
            X = self.act2b(X)
        
        # output layer
        X = self.hidden3(X)
        #X = self.act3(X)
        
        return X

    # (for trained model) given input list of 20 letter ints on [0,25], returns prob
    # that the text is true plaintext (legible) on an unlikely 0 to likely 1 scale
    def eval_text(self, text):
        
        text_input = text.copy()
        
        # make text have len 20
        if len(text_input) < 20:
            text_input.extend([26 for i in range(20 - len(text_input))])
        elif len(text_input) > 20:
            text_input = text_input[:20]
        
        char_array = np.identity(26)
        padding_array = np.zeros((26,1))
        char_array = np.hstack((char_array,padding_array))
        
        data = np.array([char_array[:,char] for char in list(text_input)], dtype=np.float32)
        data = torch.from_numpy(data)
        data = torch.unsqueeze(data, 0)
        
        self.eval()
        
        output = self(data) # evaluate input with model
        output = output.tolist()
        
        return output[0][1] # from larger output tuple, just return prob text is pt


# initialize dataset and dataloader classes
def prepare_datasets(config):
    
    # prepare the file names used to init the train/test datasets
    trainpath = "C:/Users/Tomiakfamily/Documents/Govschewlwork/2021-2022/Cryptography/Python/data/train.csv"
    testpath = "C:/Users/Tomiakfamily/Documents/Govschewlwork/2021-2022/Cryptography/Python/data/test.csv"
        
    # build the datasets
    train = PTDataset(trainpath)
    test = PTDataset(testpath)
    
    train_length = config["train_length"]
    val_length = config["val_length"]
    
    # set train/val size and build subsets from main train set
    if train_length == -1:
        train_length = int(len(train) * 0.9) # 90%/10% train/val split
    
    print('train_length:', train_length)
    
    # lists of indices used to create subset
    train_indices = range(train_length)
    val_indices = range(train_length, train_length + val_length)
    train_subset = Subset(train, train_indices)
    val_subset = Subset(train, val_indices)
    
    train_dl = DataLoader(train_subset, batch_size=config["batch_size"])
    val_dl = DataLoader(val_subset, batch_size=10, shuffle=False) # arbitrary batch size
    test_dl = DataLoader(test, batch_size=10, shuffle=False) # arbitrary batch size
    return train_dl, val_dl, test_dl


# tune the model using ray tune by training with parameters specified in tune_main()
# Access tensorboard via command line with > tensorboard --logdir=~/ray_results/my_experiment
def tune_train(config, checkpoint_dir="C:/Users/Tomiakfamily/Documents/PythonFiles/Torchtorials/checkpoints"):
    
    train_dl, val_dl, test_dl = prepare_datasets(config)

    print(len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))
    
    model = CNN(config)
    
    # set model to training mode
    model.train()
    # define the optimization
    criterion = nn.CrossEntropyLoss()
    
    if config["optim"] == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), weight_decay=config["l2_reg"])
    elif config["optim"] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    elif config["optim"] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif config["optim"] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters())
    elif config["optim"] == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())
    elif config["optim"] == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters())
    
    # variables for tracking val accuracy and implementing early stopping
    acc = 0.0
    new_acc = 0.0
    stopping_count = 0
    
    # enumerate epochs
    for epoch in range(config["epochs"]):
        print("Beginning epoch now")
        
        running_loss = 0.0
        
        model.train()
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            
            running_loss += loss.item() * inputs.size(0)
            
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
        
        with torch.no_grad():
            model.eval()
            predictions, actuals = list(), list()
            for i, (inputs, targets) in enumerate(val_dl):
                # evaluate the model on the test set
                yhat = model(inputs)
                # retrieve numpy array
                yhat = yhat.detach().numpy()
                actual = targets.numpy()
                # convert to class labels
                yhat = np.argmax(yhat, axis=1)
                # reshape for stacking
                actual = actual.reshape((len(actual), 1))
                yhat = yhat.reshape((len(yhat), 1))
                # store
                predictions.append(yhat)
                actuals.append(actual)
            predictions, actuals = vstack(predictions), vstack(actuals)
            
            old_acc = acc # remember last epoch's accuracy
            
            acc = accuracy_score(actuals, predictions)
                       
            print('Average training loss for this epoch:', running_loss/len(val_dl.dataset))
            print('accuracy for this epoch:', acc)
            
            tune.report(val_accuracy=acc, training_loss=(running_loss/len(val_dl.dataset)))
            
            # early stopping
            if (acc < old_acc) or (abs(acc-old_acc) < 0.0000001):
                stopping_count += 1
                print('Early stopping counter: ' + str(stopping_count) + '/5')
            else:
                stopping_count = 0
 
            if stopping_count == 5:
                print('Early stopping now')
                break
    
    if config["test"] == True:
        test_metrics = test(test_dl, model)
        tune.report(test_accuracy=test_metrics[0], F1=test_metrics[1])
        
        filename = "C:/Users/Tomiakfamily/Documents/Govschewlwork/2021-2022/Cryptography/Python/models/final_cnn.sav"
        torch.save(model.state_dict(), filename)


# evaluate the model
def test(test_dl, model):
    with torch.no_grad():
        model.eval()
        predictions, actuals = list(), list()
        for i, (inputs, targets) in enumerate(test_dl):
            # evaluate the model on the test set
            yhat = model(inputs)
            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = targets.numpy()
            # convert to class labels
            yhat = np.argmax(yhat, axis=1)
            # reshape for stacking
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            # store
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        # calculate accuracy
        acc = accuracy_score(actuals, predictions)
        # calculate f1score using sklearn function
        f1score = f1_score(actuals, predictions, average='binary')
    return (acc, f1score)


# used to tune model and get results for analysis with tensorboard
def tune_main():

    # hyperparameters
    config = {
        "epochs": 100,
        "lr": 0.1,
        "l2_reg": 0.0,
        "batch_size": 85,
        "num_filters": 80,
        "filter_size": 2,
        "linear_size": 190,
        "train_length": 500000, # amount of dataset used, -1 for all
        "val_length": 500, # size of validation data
        "conv_layers": 1,
        "linear_layers": 1,
        "dropout": 0.5,
        "optim": "Adam", # Adadelta, SGD, Adam, AdamW, Adagrad, RMSprop
        "init": "kaiming_uniform", # kaiming_uniform, kaiming_normal, xavier_uniform, xavier_normal, default
        "test": True,
    }
    reporter = CLIReporter(metric_columns=["training_loss", "val_accuracy", "test_accuracy", "F1"])
    scheduler = ASHAScheduler(metric="val_accuracy", mode="max")
    result = tune.run(
        tune_train,
        num_samples=1,
        config=config,
        resources_per_trial={"cpu": 3}, # 1 uses 4, 2 uses 2, 3 uses 1. Why, you may ask? I have no idea.
        # scheduler=scheduler, # commented out for tuning since disables later trials
        progress_reporter=reporter)
    
    best_trial = result.get_best_trial(metric="val_accuracy", mode="max")
    print("Best trial config: {}".format(best_trial.config))
    
    if not(config["test"]):
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["val_accuracy"]))
        
    if config["test"]:
        print("Test accuracy:", best_trial.last_result["test_accuracy"])
        print("F1 score:", best_trial.last_result["F1"])
    

if __name__ == "__main__":
    tune_main()
