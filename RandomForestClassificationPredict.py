# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 06:30:41 2020

@author: Santosh Sah
"""

import pandas as pd
from RandomForestClassificationUtils import readRandomForestClassificationModel, readRandomForestClassificationStandardScaler

def predict():
    
    randomForestClassification = readRandomForestClassificationModel()
    randomForestClassificationStandardScaler = readRandomForestClassificationStandardScaler()
    
    inputValue = [[26, 1000]]
    inputValueDataframe = pd.DataFrame(randomForestClassificationStandardScaler.transform(inputValue))
    
    predictedValue = randomForestClassification.predict(inputValueDataframe.values)
    
    print(predictedValue)

if __name__ == "__main__":
    predict()