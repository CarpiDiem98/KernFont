#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:41:43 2022

@author: emanuele
"""

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
import time

from path_font import path_kerning_file
from progettoKerning.library.dataframe import concatDataFramePlus
start_time = time.time()


# create regression data
source = '../Dataset/train/UFO'
kern_path_list = path_kerning_file(source)
letter = ['A', 'T', 'F', 'V', 'P', 'Y', 'W', 'D', 'R', 'S',
          'period', 'comma', 'v', 'f', 't', 'c', 'k', 'l', 'o', 'r', 'd', 'b']

df_complete = concatDataFramePlus(kern_path_list, letter)
df_complete = df_complete.drop(
    df_complete.columns[df_complete.apply(lambda col: col.isnull().sum() > 150)], axis=1)

x = df_complete[['peso', 'squadratura']]
# X = poly.fit_transform(X)
y = df_complete.iloc[:, 3:]
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(y)
y = imputer.transform(y)

# print(y)
np.random.seed(0)

# split into train and test data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=42)

# train the model
clf = MultiOutputRegressor(RandomForestRegressor(max_depth=2, random_state=0))
clf.fit(x_train, y_train)

# predictions
clf.predict(x_test)

score = mean_absolute_error(y_test, clf.predict(x_test))
print(score)

duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

print("--- %s minuti ---" % ((time.time() - start_time)/60))
