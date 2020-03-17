# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 06:33:09 2020

@author: Santosh Sah
"""

from RandomForestClassificationUtils import (importRandomForestClassificationDataset, saveTrainingAndTestingDataset)

def preprocess():
    
    X_train, X_test, y_train, y_test = importRandomForestClassificationDataset("Random_Forest_Classification_Social_Network_Ads.csv")
    
    saveTrainingAndTestingDataset(X_train, X_test, y_train, y_test)
    

if __name__ == "__main__":
    preprocess()