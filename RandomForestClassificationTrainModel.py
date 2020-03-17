# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 06:37:25 2020

@author: Santosh Sah
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from RandomForestClassificationUtils import (saveRandomForestClassificationModel, readRandomForestClassificationXTrain, readRandomForestClassificationYTrain,
                                     saveRandomForestClassificationStandardScaler)

"""
Train RandomForestClassification model 
"""
def trainRandomForestClassificationModel():
    
    randomForestClassificationStandardScalar = StandardScaler()
    
    X_train = readRandomForestClassificationXTrain()
    y_train = readRandomForestClassificationYTrain()
    
    randomForestClassificationStandardScalar.fit(X_train)
    saveRandomForestClassificationStandardScaler(randomForestClassificationStandardScalar)
    
    X_train = randomForestClassificationStandardScalar.transform(X_train)
    
    randomForestClassification = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 1234)
    randomForestClassification.fit(X_train, y_train)
    
    saveRandomForestClassificationModel(randomForestClassification)

if __name__ == "__main__":
    trainRandomForestClassificationModel()