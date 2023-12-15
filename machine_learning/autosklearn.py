#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 13:28:50 2022

@author: emanuele
"""

from progettoKerning.library.dataframe import concatDataFramePlus
from path_font import path_kerning_file
from numpy import random
from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error


source = '../Dataset/train/UFO'
kern_path_list = path_kerning_file(source)
letter = ['A', 'T', 'F', 'V', 'P', 'Y', 'W', 'D', 'R', 'S',
          'period', 'comma', 'v', 'f', 't', 'c', 'k', 'l', 'o', 'r', 'd', 'b']

df_complete = concatDataFramePlus(kern_path_list, letter)
X = df_complete[['peso', 'squadratura']]
y = df_complete.iloc[:, 3:].fillna(0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, shuffle=True)

model = AutoSklearnRegressor(
    time_left_for_this_task=5*60,
    per_run_time_limit=30,
    n_jobs=8)

# perform the search
model.fit(X_train, y_train)

# summarize
# print(model.sprint_statistics())

# evaluate best model
# y_hat = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_hat)
# print("MAE: %.3f" % mae)
# print(model.leaderboard())
