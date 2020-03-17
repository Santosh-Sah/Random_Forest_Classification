# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 07:30:47 2020

@author: Santosh Sah
"""

from RandomForestClassificationUtils import (readRandomForestClassificationXTest, readRandomForestClassificationModel,
                                     saveRandomForestClassificationYPred, readRandomForestClassificationStandardScaler)

"""
test the model on testing dataset
"""
def testRandomForestClassificationModel():
    
    X_test = readRandomForestClassificationXTest()
    randomForestClassificationStandardScaler = readRandomForestClassificationStandardScaler()
    X_test = randomForestClassificationStandardScaler.transform(X_test)
    
    randomForestClassificationModel = readRandomForestClassificationModel()
    
    y_pred = randomForestClassificationModel.predict(X_test)
    saveRandomForestClassificationYPred(y_pred)
    
    print(y_pred)
    
if __name__ == "__main__":
    testRandomForestClassificationModel()