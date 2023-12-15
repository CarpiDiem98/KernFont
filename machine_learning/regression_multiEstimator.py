#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:53:38 2022

@author: emanuele
"""
import os
import time
from progettoKerning.library.dataframe import concatSingleKernDataFrame, concatDataFrame, concatDataFramePlus
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from path_font import path_kerning_file
from numpy import random, sqrt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, RidgeCV
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
start_time = time.time()

source = '../Dataset/train/UFO'
kern_path_list = path_kerning_file(source)
letter = ['A', 'T', 'F', 'V', 'P', 'Y', 'W', 'D', 'R', 'S',
          'period', 'comma', 'v', 'f', 't', 'c', 'k', 'l', 'o', 'r', 'd', 'b']

df_complete = concatDataFramePlus(kern_path_list, letter)
# df = concatSingleKernDataFrame(kern_path_list, 'A', 'T')
# df_complete = df_complete.drop(
# df_complete.columns[df_complete.apply(lambda col: col.isnull().sum() > 150)], axis=1)

# poly = PolynomialFeatures(2)
X = df_complete[['peso', 'squadratura']]
# X = poly.fit_transform(X)
y = df_complete.iloc[:, 3:].fillna(0)
# X = df.iloc[:, 1:5]
# X = poly.fit_transform(X)
# y = df['kern'].fillna(0)
# print(y)
random.seed(0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.fit(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, shuffle=True)


def show_image(X_test, y_test_predict, name):
    # scattor plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_predict[name], y_test, cmap='plasma')
    plt.title(name)
    plt.show()
    print('MAE for ', name, ' is ', y_mae[name])
    # to plot the faces
    image_shape = (64, 64)
    plt.figure(figsize=(10, 10))
# plt.show()


ESTIMATORS = {
    # "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       # max_features=32,     # Out of 20000
                                       # random_state=0),
    # Accept default parameters
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(random_state=0),
    "RandomForestRegressor": RandomForestRegressor(max_depth=4, random_state=2),
    "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5),
    # "MultiO/P GBR": GradientBoostingRegressor(n_estimators=5),
    # "MultiO/P AdaB": AdaBoostRegressor(n_estimators=5),
    "MultiLayer Perception Regressor": MLPRegressor(random_state=0, max_iter=1000)
}

y_test_predict = dict()
y_mae = dict()

for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)

    y_test_predict[name] = estimator.predict(X_test)
    y_mae[name] = mean_absolute_error(y_test, estimator.predict(X_test))
    show_image(X_test, y_test_predict, name)


###############################################################################
duration = 1  # seconds
freq = 440  # Hz
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

print("--- %s minuti ---" % ((time.time() - start_time)/60))
