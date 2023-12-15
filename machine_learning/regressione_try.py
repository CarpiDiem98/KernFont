#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 16:29:37 2022

@author: emanuele
"""
from progettoKerning.library.dataframe import concatSingleKernDataFrame
import pandas as pd
from path_font import path_kerning_file
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import random

source = '../Dataset/train/UFO'
kern_path_list = path_kerning_file(source)
# df = concatSingleKernDataFrame(kern_path_list, 'A', 'T')

# poly = PolynomialFeatures(2)

X = df.iloc[:,1:5]
# X = poly.fit_transform(X)
y = df['kern'].fillna(0)
print(y)
random.seed(0)

# sns.regplot(x="peso", y="squadratura", data=df, fit_reg=False)

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lm = LinearRegression()
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=5)
# print(scores)

# can tune other metrics, such as MSE
scores = cross_val_score(lm, X_train, y_train,
                         scoring='neg_mean_squared_error', cv=5, error_score='raise')
# print(scores)

# step-1: create a cross-validation scheme
folds = KFold(n_splits=5, shuffle=True, random_state=100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 5))}]

# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator=rfe,
                        param_grid=hyper_params,
                        scoring='r2',
                        cv=folds,
                        verbose=1,
                        return_train_score=True)

# fit the model
model_cv.fit(X_train, y_train)

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# plotting cv results
plt.figure(figsize=(16, 6))

plt.plot(cv_results["param_n_features_to_select"],
         cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"],
         cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# model = Pipeline([('Poly', PolynomialFeatures(degree=3)),
# ('Linear', LinearRegression())])

# cv = KFold(n_splits=10, random_state=0, shuffle=True)
# model = LinearRegression()
# scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
# mean(absolute(scores))

# model.fit(X_train, y_train)
# predict = model.predict(X_test)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# regression = LinearRegression()
# regression.fit(X_train, y_train)
# predict = regression.predict(X_test)

# # errore medio 5.qualcosa
# raw = mean_absolute_error(y_test, predict, multioutput='raw_values')
# value = mean_absolute_error(y_test, predict)
# print(raw, value)
