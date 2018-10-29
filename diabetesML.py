# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:46:40 2018

@author: TEEE
"""

"""
Created on Fri Apr 20 19:43:17 2018

@author: TEEE
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def clean(y_pred):
    for i in range(len(y_pred)):
        if (y_pred[i]>=0.5):
            y_pred[i]=1
        else:
            y_pred[i]=0
    return y_pred

def compare(y_pred,y_test):
    a=[]
    for i in range(len(y_pred)):
        if (y_pred[i]==y_test[i]):
            a.append('true')
        else:
            a.append('false')
    return a       

def percen(a):
    tr=0
    fl=0
    for i in range(len(a)):
        if (a[i]=='true'):
            tr+=1
        else:
            fl+=1
    print('true=',tr,', false',fl)
    print((tr/(fl+tr))*100, '%')

# Importing the dataset
dataset = pd.read_csv('pima-indians-diabetes.csv')
X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8:9].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_train_svr=y_train
y_test_svr=y_test
y_train = y_train[:, 0]
y_test = y_test[:, 0]
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
# multiple linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred1 = regressor.predict(X_test)

#testing SVR

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_svr = sc_X.fit_transform(X_train)
y_svr = sc_y.fit_transform(y_train_svr)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor1 = SVR(kernel = 'rbf')
regressor1.fit(X_svr, y_svr)

# Predicting a new result
y_pred2 = sc_y.inverse_transform( regressor1.predict(sc_X.transform(X_test)))

#cleaning up mess
y_pred1=clean(y_pred1)
y_pred2=clean(y_pred2)

mult=compare(y_pred1,y_test)
svr=compare(y_pred2,y_test)
#show results
percen(mult)
percen(svr)