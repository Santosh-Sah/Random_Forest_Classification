# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 07:43:04 2020

@author: Santosh Sah
"""

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from RandomForestClassificationUtils import (readRandomForestClassificationYTest, readRandomForestClassificationYPred)

"""

calculating RandomForestClassification confussion matrix

"""
def testRandomForestClassificationConfussionMatrix():
    
    y_test = readRandomForestClassificationYTest()
    y_pred = readRandomForestClassificationYPred()
    
    randomForestClassificationConfussionMatrix = confusion_matrix(y_test, y_pred)
    print(randomForestClassificationConfussionMatrix)
    
    """
    Below is the confussion matrix
    [[57  1]
    [ 3 19]]
    
    """
"""
calculating accuracy score

"""

def testRandomForestClassificationAccuracy():
    
    y_test = readRandomForestClassificationYTest()
    y_pred = readRandomForestClassificationYPred()
    
    randomForestClassificationConfussionAccuracy = accuracy_score(y_test, y_pred)
    
    print(randomForestClassificationConfussionAccuracy) #.95%

"""
calculating classification report

"""

def testRandomForestClassificationClassificationReport():
    
    y_test = readRandomForestClassificationYTest()
    y_pred = readRandomForestClassificationYPred()
    
    randomForestClassificationConfussionClassificationReport = classification_report(y_test, y_pred)
    
    print(randomForestClassificationConfussionClassificationReport)
    
    """
             precision    recall  f1-score   support

          0       0.95      0.98      0.97        58
          1       0.95      0.86      0.90        22

avg / total       0.95      0.95      0.95        80
    """
    
if __name__ == "__main__":
    #testRandomForestClassificationConfussionMatrix()
    #testRandomForestClassificationAccuracy()
    testRandomForestClassificationClassificationReport()