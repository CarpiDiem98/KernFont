#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:11:07 2022

@author: emanuele
"""

from progettoKerning.library.dataframe import concatDataFrame, concatDataFramePlus
from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import LinearSVR
from path_font import path_kerning_file
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.model_selection import cross_val_score, RepeatedKFold, train_test_split
from numpy import mean, std

letter = ['A', 'T', 'F', 'V', 'P', 'Y', 'W', 'D', 'R', 'S',
          'period', 'comma', 'v', 'f', 't', 'c', 'k', 'l', 'o', 'r', 'd', 'b']

source = '../Dataset/train/UFO'
kern_path_list = path_kerning_file(source)

# df_plus = concatDataFramePlus(kern_path_list, letter)
df_plus = df_plus.drop(
    df_plus.columns[df_plus.apply(lambda col: col.isnull().sum() > 150)], axis=1)

df_plus = df_plus.loc[~(df_plus.isna().sum(axis=1)>=30)]

X = df_plus[['peso', 'squadratura']]
y = df_plus.iloc[:, 3:].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, shuffle=True)

# define base model
model = LinearRegression()
# define the direct multioutput wrapper model
wrapper = RegressorChain(model)
# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=0)
# evaluate the model and collect the scores
n_scores = cross_val_score(wrapper, X_train, y_train,
                           scoring='neg_mean_absolute_error', cv=cv)
# force the scores to be positive
n_scores = abs(n_scores)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

wrapper.fit(X_train, y_train)
wrapper.predict(X_test)


def mae_cv(model):
    rmae = abs(
        (cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=5)).mean())
    return rmae


mae_cross_val = mae_cv(wrapper)
mae_cross_val
