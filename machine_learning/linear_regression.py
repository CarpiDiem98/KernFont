#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 16:46:21 2022

@author: emanuele
"""

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from numpy import random
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
from progettoKerning.library.dataframe import concatDataFrame

df = concatDataFrame()
# print(df)

X = df[['peso', 'squadratura']]
y = df.iloc[:, 3:]
random.seed(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

regression = LinearRegression()
regression.fit(X_train, y_train)
predict = regression.predict(X_test)

# errore medio 5.qualcosa
raw = mean_absolute_error(y_test, predict, multioutput='raw_values')
value = mean_absolute_error(y_test, predict)
# print(raw, '\n', value)


# df[df['squadratura'] < 0.5]  # 11 righe
# df[df['squadratura'] > 1] # 3 righe

