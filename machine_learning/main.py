#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 19:14:07 2022

@author: emanuele
"""
from progettoKerning.library.dataframe import concatDataFrame
from sklearn.linear_model import LinearRegression
from re import search
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


df = concatDataFrame()
# qui abbiamo solo le font di cast in teoria è una matrice starsa della stra madonna troia
# df = concatDataFrame()

X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

scaler = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions_test = lin_reg.predict(X_test)
prediction_train = lin_reg.predict(X_train)

def mae_cv(model):
    rmae = abs(
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

print('------------------------- test set evaluation -------------------------')
mae, rmse, r_squared, mae_cross_val = print_evaluate(y_test, predictions_test, lin_reg)

new_row = {"Model": "LinearRegression", "MAE": mae, "RMSE": rmse,
           "R2 Score": r_squared, "MAE (Cross-Validation)": mae_cross_val}

print('------------------------- train set evaluation -------------------------')
mae, rmse, r_squared, mae_cross_val = print_evaluate(y_train, prediction_train, lin_reg)
print('------------------------------------------------------------------------')

