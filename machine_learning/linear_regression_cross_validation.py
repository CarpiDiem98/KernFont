#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:52:05 2022

@author: emanuele
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
# from numpy import random
from sklearn.model_selection import cross_val_score, train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from progettoKerning.library.dataframe import concatDataFrame
from numpy import absolute, std
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.pipeline import Pipeline


df = concatDataFrame()
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.iloc[:, 1:].corr(), cmap="RdBu")
# plt.title("Correlations Between Variables", size=15)
# plt.show()

X = df[['peso', 'squadratura']]
y = df.iloc[:, 3:]

scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def mae_cv(model):
    rmae = absolute(
        (cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=5)).mean())
    return rmae


def print_evaluate(y, predictions, lin_reg):  
    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    mae_cross_val = mae_cv(lin_reg)
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('R2 Score', r_squared)
    print("MAE Cross-Validation:", mae_cross_val)
    print("-"*30)
    return mae, rmse, r_squared, mae_cross_val

models = pd.DataFrame(
    columns=["Model", "MAE", "RMSE", "R2 Score", "MAE (Cross-Validation)"])

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
predictions_test = lin_reg.predict(X_test)
prediction_train = lin_reg.predict(X_train)

print('------------------------- test set evaluation -------------------------')
mae, rmse, r_squared, mae_cross_val = print_evaluate(y_test, predictions_test, lin_reg)

new_row = {"Model": "LinearRegression", "MAE": mae, "RMSE": rmse,
           "R2 Score": r_squared, "MAE (Cross-Validation)": mae_cross_val}
models = models.append(new_row, ignore_index=True)

print('------------------------- train set evaluation -------------------------')
mae, rmse, r_squared, mae_cross_val = print_evaluate(y_train, prediction_train, lin_reg)
print('------------------------------------------------------------------------')

###############################################################################
gradi = [2, 3, 4, 5, 6]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

train_X = np.asanyarray(X_train)
train_y = np.asanyarray(y_train)

test_X = np.asanyarray(X_test)
test_y = np.asanyarray(y_test)

for degree in gradi:
    poly = PolynomialFeatures(degree=degree)
    train_X_poly = poly.fit_transform(train_X)
    test_X_poly = poly.fit_transform(test_X)

    lin_reg = LinearRegression()
    train_y_ = lin_reg.fit(train_X_poly, train_y)
    predictions = lin_reg.predict(test_X_poly)
    
    print('Polynomial LR:', degree)
    mae, rmse, r_squared, mae_cross_val = print_evaluate(test_y, predictions, lin_reg)
    new_row = {"Model": "Polynomial " + str(degree) + " LR", "MAE": mae, "RMSE": rmse,
                "R2 Score": r_squared, "MAE (Cross-Validation)": mae_cross_val}
    # print('Polynomial LR:', degree)
    # print("MAE:", mae)
    # print("RMSE:", rmse)
    # print("R2 Score:", r_squared)
    mae_cross_val = mae_cv(lin_reg)
    print("MAE Cross-Validation:", mae_cross_val)
    # print("-"*30)
    # models = models.append(new_row, ignore_index=True)










