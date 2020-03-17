# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 06:42:59 2020

@author: Santosh Sah
"""

"""
importing the libraries
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importRandomForestClassificationDataset(randomForestClassificationDatasetFileName):
    
    randomForestClassificationDataset = pd.read_csv(randomForestClassificationDatasetFileName)
    X = randomForestClassificationDataset.iloc[:, [2, 3]].values
    y = randomForestClassificationDataset.iloc[:, 4].values
    
    #spliting the dataset into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    return X_train, X_test, y_train, y_test

"""
Save standard scalar object as a pickel file. This standard scalar object must be used to standardized the dataset for training, testing and new dataset.
To use this standard scalar object we need to read it and then use it.
"""
def saveRandomForestClassificationStandardScaler(randomForestClassificationStandardScalar):
    
    #Write RandomForestClassificationStandardScaler in a picke file
    with open("RandomForestClassificationStandardScaler.pkl",'wb') as RandomForestClassificationStandardScaler_Pickle:
        pickle.dump(randomForestClassificationStandardScalar, RandomForestClassificationStandardScaler_Pickle, protocol = 2)

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)
    
    #Write y_train in a picke file
    with open("y_train.pkl",'wb') as y_train_Pickle:
        pickle.dump(y_train, y_train_Pickle, protocol = 2)
    
    #Write y_test in a picke file
    with open("y_test.pkl",'wb') as y_test_Pickle:
        pickle.dump(y_test, y_test_Pickle, protocol = 2)

"""
Save RandomForestClassificationModel as a pickle file.
"""
def saveRandomForestClassificationModel(randomForestClassificationModel):
    
    #Write RandomForestClassificationModel as a picke file
    with open("RandomForestClassificationModel.pkl",'wb') as RandomForestClassificationModel_Pickle:
        pickle.dump(randomForestClassificationModel, RandomForestClassificationModel_Pickle, protocol = 2)

"""
read RandomForestClassificationStandardScalar from pickel file
"""
def readRandomForestClassificationStandardScaler():
    
    #load RandomForestClassificationStandardScaler object
    with open("RandomForestClassificationStandardScaler.pkl","rb") as RandomForestClassificationStandardScaler:
        randomForestClassificationStandardScalar = pickle.load(RandomForestClassificationStandardScaler)
    
    return randomForestClassificationStandardScalar

"""
read RandomForestClassificationModel from pickle file
"""
def readRandomForestClassificationModel():
    
    #load RandomForestClassificationModel model
    with open("RandomForestClassificationModel.pkl","rb") as RandomForestClassificationModel:
        randomForestClassificationModel = pickle.load(RandomForestClassificationModel)
    
    return randomForestClassificationModel

"""
read X_train from pickle file
"""
def readRandomForestClassificationXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readRandomForestClassificationXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
read y_train from pickle file
"""
def readRandomForestClassificationYTrain():
    
    #load y_train
    with open("y_train.pkl","rb") as y_train_pickle:
        y_train = pickle.load(y_train_pickle)
    
    return y_train

"""
read y_test from pickle file
"""
def readRandomForestClassificationYTest():
    
    #load y_test
    with open("y_test.pkl","rb") as y_test_pickle:
        y_test = pickle.load(y_test_pickle)
    
    return y_test

"""
save y_pred as a pickle file
"""

def saveRandomForestClassificationYPred(y_pred):
    
    #Write y_red in a picke file
    with open("y_pred.pkl",'wb') as y_pred_Pickle:
        pickle.dump(y_pred, y_pred_Pickle, protocol = 2)

"""
read y_predt from pickle file
"""
def readRandomForestClassificationYPred():
    
    #load y_test
    with open("y_pred.pkl","rb") as y_pred_pickle:
        y_pred = pickle.load(y_pred_pickle)
    
    return y_pred